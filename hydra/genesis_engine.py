"""
Hydra Genesis Engine — Multi-Agent Swarm Orchestrator
=======================================================
The top-level orchestrator for the Genesis architecture.

Pipeline per bar:
  MarketDataBus → [L2Feed + ContextFeed + Heatmap]
      ↓
  FractionalDifferentiator (stationarity)
      ↓
  ChartAgent (PatchTST) ──┐
  RLAgent    (PPO)       ──┼→ HeadAgent (Consensus) → TradeStruct
  LLMAgent   (Claude)   ──┘
      ↓
  SLTPEngine + FeeGuard + LeverageController
      ↓
  Execution (paper / live)

State Machine:
  IDLE → WARMING_UP → ANALYZING → SIGNAL_FOUND → EXECUTING → MONITORING → IDLE
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


import yaml

from hydra.data_engineer.market_bus import MarketDataBus, MarketEvent, EventType, InProcessBus
from hydra.data_engineer.frac_diff import FractionalDifferentiator
from hydra.data_engineer.heatmap import LiquidityHeatmap, WhaleClusters
from hydra.data_engineer.context_feed import ContextFeed, ContextSnapshot
from hydra.specialist_agents.chart_agent import ChartAgent, ChartSignal
from hydra.specialist_agents.rl_agent import RLAgent, RLDecision
from hydra.specialist_agents.llm_agent import LLMAgent, LLMSignal
from hydra.head_agent.signal_fusion import HeadAgent, TradeStruct

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  State Machine
# ─────────────────────────────────────────────

class GenesisState(str, Enum):
    IDLE        = "IDLE"
    WARMING_UP  = "WARMING_UP"
    ANALYZING   = "ANALYZING"
    SIGNAL_FOUND= "SIGNAL_FOUND"
    EXECUTING   = "EXECUTING"
    MONITORING  = "MONITORING"
    STOPPED     = "STOPPED"


@dataclass
class GenesisTrade:
    """A trade record produced by the Genesis engine."""
    trade_id: int
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_usd: float
    leverage: float
    kelly_fraction: float
    entry_time_ns: int
    trade_struct: Optional[TradeStruct] = None
    exit_price: Optional[float] = None
    exit_time_ns: Optional[int] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.exit_price is None


@dataclass
class GenesisStats:
    bars_processed: int = 0
    signals_generated: int = 0
    signals_vetoed: int = 0
    signals_fee_rejected: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    llm_calls: int = 0

    @property
    def win_rate(self) -> float:
        return self.winning_trades / max(self.total_trades, 1)

    @property
    def profit_factor(self) -> float:
        """Requires trade_history — computed externally."""
        return 0.0


# ─────────────────────────────────────────────
#  Genesis Engine
# ─────────────────────────────────────────────

class GenesisEngine:
    """
    Hydra Multi-Agent Genesis Engine.

    Usage:
        engine = GenesisEngine.from_config('config/genesis_config.yaml')
        asyncio.run(engine.run(mode='backtest', n_bars=10000))
    """

    def __init__(self, config: dict, mode: str = "backtest") -> None:
        self.config = config
        self.mode = mode
        self.state = GenesisState.IDLE
        self.stats = GenesisStats()

        # ── Market Data Bus ────────────────────
        self.bus: InProcessBus = MarketDataBus.from_config(config)

        # ── Data Layer ────────────────────────
        self.frac_diff = FractionalDifferentiator(
            d=config.get("frac_diff", {}).get("d", 0.4),
            threshold=config.get("frac_diff", {}).get("threshold", 1e-5),
        )
        self.heatmap = LiquidityHeatmap(config, bus=self.bus)
        self.context_feed = ContextFeed(config, bus=self.bus, mode=mode)

        # ── Specialist Agents ──────────────────
        self.chart_agent = ChartAgent(
            config,
            model_path=config.get("chart_agent", {}).get("model_path"),
            confidence_thr=config.get("chart_agent", {}).get("confidence_thr", 0.60),
        )
        self.rl_agent = RLAgent(
            config,
            model_path=config.get("rl_agent", {}).get("model_path"),
        )
        self.llm_agent = LLMAgent(
            config,
            mode=mode,
            min_interval_s=config.get("llm", {}).get("min_interval_s", 60.0),
        )

        # ── Head Agent ────────────────────────
        ha_cfg = config.get("head_agent", {})
        self.head_agent = HeadAgent(
            config,
            weight_chart=ha_cfg.get("weight_chart", 0.35),
            weight_rl=ha_cfg.get("weight_rl", 0.40),
            weight_llm=ha_cfg.get("weight_llm", 0.25),
            min_consensus_score=ha_cfg.get("min_consensus_score", 0.25),
        )

        # ── Risk Layer ────────────────────────
        # Reuse existing risk modules from the Automated-Technical-Analyst package
        try:
            from hydra.risk_manager.sl_tp_engine import SLTPEngine
            from hydra.risk_manager.fee_guard import FeeGuard
            self.sl_tp_engine = SLTPEngine(config)
            self.fee_guard = FeeGuard(config)
            self._has_risk = True
        except ImportError:
            logger.warning("Risk modules not found — using simplified risk logic.")
            self.sl_tp_engine = None
            self.fee_guard = None
            self._has_risk = False

        # ── Account State ─────────────────────
        self._balance: float = float(config.get("risk", {}).get("initial_balance", 1000.0))
        self._peak_balance = self._balance
        self._current_trade: Optional[GenesisTrade] = None
        self._trade_history: List[GenesisTrade] = []
        self._trade_id = 0
        self._last_context: Optional[ContextSnapshot] = None
        self._last_clusters: Optional[WhaleClusters] = None

        # ── Momentum Tracker ──────────────────
        # Rolling 20-bar price buffer for momentum gate
        self._price_hist: List[float] = []
        self._MOMENTUM_WINDOW = 20  # bars
        self._MOMENTUM_THRESH = 0.001  # 0.1% in 20 bars (more sensitive)

        import numpy as np
        self._np = np

        logger.info(
            "GenesisEngine | mode=%s | balance=%.2f | bus=%s",
            mode, self._balance, type(self.bus).__name__,
        )

    @classmethod
    def from_config(cls, config_path: str = "config/genesis_config.yaml", mode: str = "backtest") -> "GenesisEngine":
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg, mode)

    # ── Main Entry Point ──────────────────────

    async def run(self, n_bars: Optional[int] = None) -> GenesisStats:
        """
        Run the Genesis Engine.

        Args:
            n_bars: Optional bar limit (for backtesting).

        Returns:
            Final GenesisStats.
        """
        self.state = GenesisState.WARMING_UP
        await self.bus.connect()
        logger.info("🌀 Hydra Genesis Engine — SWARM ACTIVATED")

        # Start context feed in background
        context_task = asyncio.create_task(self.context_feed.start())

        # Build synthetic or live L2 feed
        feed = self._build_feed()
        feed_task = asyncio.create_task(feed.start())

        try:
            await self._main_loop(feed, n_bars)
        finally:
            await feed.stop()
            await self.context_feed.stop()
            try:
                await asyncio.wait_for(feed_task, timeout=5.0)
                await asyncio.wait_for(context_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
            self.state = GenesisState.STOPPED
            self._log_final_stats()

        return self.stats

    def _build_feed(self):
        """Build appropriate feed for current mode."""
        if self.mode == "backtest":
            # Use existing SyntheticL2Feed
            try:
                from hydra.data_engineer.l2_feed import SyntheticL2Feed
                return SyntheticL2Feed(self.config)
            except ImportError:
                from hydra.data_engineer.l2_feed_genesis import L2FeedGenesis
                return L2FeedGenesis(self.config, bus=self.bus)
        else:
            from hydra.data_engineer.l2_feed_genesis import L2FeedGenesis
            return L2FeedGenesis(self.config, bus=self.bus)

    # ── Core Loop ─────────────────────────────

    async def _main_loop(self, feed, n_bars: Optional[int]) -> None:
        bars = 0
        while self.state != GenesisState.STOPPED:
            if n_bars and bars >= n_bars:
                break
            try:
                snapshot = await asyncio.wait_for(feed.queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                if self.mode == "backtest" and not feed._running:
                    break
                logger.warning("Genesis: feed timeout.")
                continue

            bars += 1
            self.stats.bars_processed = bars
            self.state = GenesisState.ANALYZING

            await self._process_bar(snapshot)

        self.state = GenesisState.STOPPED

    async def _process_bar(self, snapshot) -> None:
        """Full per-bar processing pipeline."""
        mid = snapshot.mid_price
        ts_ns = snapshot.timestamp_ns
        np = self._np

        # ── 1. Momentum tracker ───────────────
        self._price_hist.append(mid)
        if len(self._price_hist) > self._MOMENTUM_WINDOW + 5:
            self._price_hist = self._price_hist[-(self._MOMENTUM_WINDOW + 5):]

        momentum = 0.0
        if len(self._price_hist) >= self._MOMENTUM_WINDOW:
            momentum = (mid - self._price_hist[-self._MOMENTUM_WINDOW]) / max(self._price_hist[-self._MOMENTUM_WINDOW], 1)

        # ── 2. Fractional Differentiation ─────
        frac_price = self.frac_diff.update(mid) or 0.0

        # ── 3. Heatmap Update ─────────────────
        bids = [(lvl.price, lvl.quantity) for lvl in snapshot.bids]
        asks = [(lvl.price, lvl.quantity) for lvl in snapshot.asks]
        clusters = self.heatmap.update(bids, asks, mid, ts_ns)
        self._last_clusters = clusters
        if self.stats.bars_processed % 20 == 0:
            await self.heatmap.publish(clusters)

        # ── 4. Context ────────────────────────
        ctx = self.context_feed.last_snapshot
        self._last_context = ctx
        funding_rate = ctx.funding_rate if ctx else 0.0
        oi_delta     = ctx.oi_delta if ctx else 0.0
        est_liq      = ctx.estimated_liq if ctx else 0.0
        funding_sentiment = ctx.sentiment if ctx else "NEUTRAL"

        # ── 5. LOB Features ───────────────────
        # Compute OFI from top-10 LOB levels (bid notional - ask notional)
        top_bids = bids[:10]
        top_asks = asks[:10]
        total_bid = sum(p * q for p, q in top_bids)
        total_ask = sum(p * q for p, q in top_asks)
        total_bid_full = sum(p * q for p, q in bids)
        total_ask_full = sum(p * q for p, q in asks)
        book_pressure = total_bid_full / max(total_bid_full + total_ask_full, 1e-8)
        # OFI: normalised bid-ask imbalance in top 10 levels
        ofi = (total_bid - total_ask) / max(total_bid + total_ask, 1e-8)
        spread_bps = ((snapshot.best_ask - snapshot.best_bid) / max(mid, 1)) * 10000
        price_to_vwap = 0.0

        # ── 6. Monitor Open Trade ─────────────
        if self._current_trade is not None:
            await self._monitor_trade(mid, ts_ns)
            if self._current_trade is not None:
                return

        # ── 7. Specialist Agents ──────────────
        chart_signal = self.chart_agent.process(
            mid_price=mid, feature_vec=None, timestamp_ns=ts_ns,
        )

        rl_decision = self.rl_agent.process(
            ofi=ofi, book_pressure=book_pressure, spread_bps=spread_bps,
            price_to_vwap=price_to_vwap, funding_rate=funding_rate,
            oi_delta=oi_delta, est_liq=est_liq, frac_diff_price=frac_price,
            running_pnl=self.stats.total_pnl, drawdown=self.stats.max_drawdown,
            win_rate=self.stats.win_rate, balance=self._balance,
            timestamp_ns=ts_ns,
        )

        llm_signal = await self.llm_agent.process(
            mid_price=mid,
            bid_wall=getattr(snapshot, 'bid_wall_price', snapshot.best_bid),
            ask_wall=getattr(snapshot, 'ask_wall_price', snapshot.best_ask),
            cum_bid_depth=total_bid_full, cum_ask_depth=total_ask_full,
            support_levels=clusters.support_levels[:3],
            resistance_levels=clusters.resistance_levels[:3],
            funding_rate=funding_rate,
            open_interest=ctx.open_interest if ctx else 1e9,
            oi_delta=oi_delta,
            patterns=chart_signal.patterns if chart_signal else [],
            rl_direction=rl_decision.direction if rl_decision else "HOLD",
            timestamp_ns=ts_ns,
        )

        # ── 8. Head Agent Fusion ──────────────
        depth_ratio = total_bid_full / max(total_ask_full, 1)
        trade_struct = self.head_agent.fuse(
            chart=chart_signal, rl=rl_decision, llm=llm_signal,
            mid_price=mid, heatmap_depth_ratio=depth_ratio,
            funding_sentiment=funding_sentiment, timestamp_ns=ts_ns,
        )

        # ── 9. Momentum Gate ──────────────────
        # Only allow trades when 20-bar momentum confirms signal direction.
        # This is the primary WR improvement: enter only WITH the trend.
        if trade_struct.is_actionable and not trade_struct.vetoed:
            direction = trade_struct.direction
            momentum_ok = (
                (direction == "LONG"  and momentum >= self._MOMENTUM_THRESH) or
                (direction == "SHORT" and momentum <= -self._MOMENTUM_THRESH) or
                abs(momentum) < 1e-6  # not enough history yet — allow through
            )
            if not momentum_ok:
                trade_struct = dataclasses.replace(
                    trade_struct, vetoed=True, veto_reason="MOMENTUM_GATE"
                )

        await self.bus.publish(MarketEvent(
            type=EventType.SIGNAL, data=trade_struct.to_json(), source="head_agent",
        ))

        if trade_struct.is_actionable:
            self.stats.signals_generated += 1
            if trade_struct.vetoed:
                self.stats.signals_vetoed += 1
            else:
                self.state = GenesisState.SIGNAL_FOUND
                await self._evaluate_and_execute(trade_struct, snapshot)
        elif trade_struct.vetoed:
            self.stats.signals_vetoed += 1

    # ── Trade Lifecycle ───────────────────────

    async def _evaluate_and_execute(self, ts: TradeStruct, snapshot) -> None:
        mid = ts.mid_price
        direction = ts.direction

        # Fixed ATR-style SL/TP: SL=0.4%, TP=1.0% → RR = 2.5 always
        # This avoids the old leverage-scaled TP that created nonsensical RR values.
        SL_PCT = 0.004   # 0.4% stop loss
        TP_PCT = 0.010   # 1.0% take profit (2.5× RR)
        if direction == "LONG":
            sl = mid * (1 - SL_PCT)
            tp = mid * (1 + TP_PCT)
        else:
            sl = mid * (1 + SL_PCT)
            tp = mid * (1 - TP_PCT)

        leverage = float(min(ts.rl_leverage, self.config.get("leverage", {}).get("tier_3_leverage", 10)))
        kelly = float(ts.rl_kelly)
        position_usd = self._balance * kelly * leverage

        # Fee gate: expected gross profit must exceed 3× round-trip fees
        fee_rate = self.config.get("risk", {}).get("fee_rate_taker", 0.0004)
        round_trip_fees = 2 * position_usd * fee_rate
        expected_profit = TP_PCT * position_usd
        if expected_profit < round_trip_fees * 3:
            self.stats.signals_fee_rejected += 1
            return

        # Open trade
        self._trade_id += 1
        trade = GenesisTrade(
            trade_id=self._trade_id,
            direction=direction,
            entry_price=mid,
            stop_loss=sl,
            take_profit=tp,
            position_usd=position_usd,
            leverage=leverage,
            kelly_fraction=kelly,
            entry_time_ns=ts.timestamp_ns,
            trade_struct=ts,
        )
        self._current_trade = trade
        self.state = GenesisState.MONITORING
        logger.info(
            "📈 Genesis Trade #%d OPEN | %s @ %.2f | SL=%.2f | TP=%.2f | $%.2f | %dx",
            trade.trade_id, direction, mid, sl, tp, position_usd, int(leverage),
        )

    async def _monitor_trade(self, mid: float, ts_ns: int) -> None:
        trade = self._current_trade
        if not trade:
            return

        hit_sl = (trade.direction == "LONG" and mid <= trade.stop_loss) or \
                 (trade.direction == "SHORT" and mid >= trade.stop_loss)
        hit_tp = (trade.direction == "LONG" and mid >= trade.take_profit) or \
                 (trade.direction == "SHORT" and mid <= trade.take_profit)

        if hit_sl:
            await self._close_trade(trade.stop_loss, "SL", ts_ns)
        elif hit_tp:
            await self._close_trade(trade.take_profit, "TP", ts_ns)

    async def _close_trade(self, exit_price: float, reason: str, ts_ns: int) -> None:
        trade = self._current_trade
        if not trade:
            return

        trade.exit_price = exit_price
        trade.exit_time_ns = ts_ns
        trade.exit_reason = reason

        price_move = exit_price - trade.entry_price
        if trade.direction == "SHORT":
            price_move = -price_move
        units = trade.position_usd / trade.entry_price
        gross_pnl = price_move * units
        fee_rate = self.config.get("risk", {}).get("fee_rate_taker", 0.0004)
        fees = 2 * trade.position_usd * fee_rate
        trade.pnl = gross_pnl - fees

        self._balance += trade.pnl
        self.stats.total_pnl += trade.pnl
        self.stats.total_trades += 1
        if trade.pnl > 0:
            self.stats.winning_trades += 1

        self._peak_balance = max(self._peak_balance, self._balance)
        dd = (self._peak_balance - self._balance) / self._peak_balance
        self.stats.max_drawdown = max(self.stats.max_drawdown, dd)

        self.rl_agent.record_trade_pnl(trade.pnl)

        emoji = "✅" if trade.pnl > 0 else "❌"
        logger.info(
            "%s Genesis Trade #%d CLOSED | %s @ %.2f | PnL=$%.4f | balance=$%.2f",
            emoji, trade.trade_id, reason, exit_price, trade.pnl, self._balance,
        )

        self._trade_history.append(trade)
        self._current_trade = None
        self.state = GenesisState.ANALYZING

    # ── Reporting ─────────────────────────────

    def _log_final_stats(self) -> None:
        s = self.stats
        wins = [t.pnl for t in self._trade_history if t.pnl and t.pnl > 0]
        losses = [abs(t.pnl) for t in self._trade_history if t.pnl and t.pnl < 0]
        pf = sum(wins) / max(sum(losses), 1e-8)

        logger.info("=" * 60)
        logger.info("🌀 HYDRA GENESIS ENGINE — FINAL REPORT")
        logger.info("  Bars processed:     %d", s.bars_processed)
        logger.info("  Signals generated:  %d", s.signals_generated)
        logger.info("  Signals vetoed:     %d", s.signals_vetoed)
        logger.info("  Fee rejections:     %d", s.signals_fee_rejected)
        logger.info("  Total trades:       %d", s.total_trades)
        logger.info("  Win rate:           %.2f%%", s.win_rate * 100)
        logger.info("  Profit Factor:      %.3f", pf)
        logger.info("  Total PnL:          $%.4f", s.total_pnl)
        logger.info("  Max drawdown:       %.2f%%", s.max_drawdown * 100)
        logger.info("  Final balance:      $%.2f", self._balance)
        logger.info("  Head Agent veto %%: %.1f%%", self.head_agent.stats["veto_rate"] * 100)
        logger.info("=" * 60)

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def trade_history(self) -> List[GenesisTrade]:
        return self._trade_history.copy()
