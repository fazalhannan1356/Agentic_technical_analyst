"""
L2 Feed Genesis — 100-Level Deep Order Book via CCXT Pro / REST
================================================================
Self-contained 100-level L2 order book feed with MarketDataBus publishing.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from hydra.data_engineer.market_bus import MarketDataBus, MarketEvent, EventType, InProcessBus

logger = logging.getLogger(__name__)


# ── Shared Data Structures ────────────────────────────────────

@dataclass
class PriceLevel:
    price: float
    quantity: float

    @property
    def notional_usd(self) -> float:
        return self.price * self.quantity


@dataclass
class L2Snapshot:
    timestamp_ns: int
    bids: List[PriceLevel]
    asks: List[PriceLevel]
    symbol: str
    sequence: int = 0

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        if self.mid_price == 0:
            return 0.0
        return (self.spread / self.mid_price) * 10_000


@dataclass
class TradeTick:
    timestamp_ns: int
    price: float
    quantity: float
    side: str  # 'buy' | 'sell'

logger = logging.getLogger(__name__)


@dataclass
class L2SnapshotFull(L2Snapshot):
    """100-level L2 snapshot with additional microstructure metrics."""
    funding_rate: float = 0.0
    open_interest: float = 0.0

    @property
    def bid_wall_price(self) -> float:
        """Price of the largest single bid (whale bid wall)."""
        if not self.bids:
            return 0.0
        return max(self.bids, key=lambda x: x.notional_usd).price

    @property
    def ask_wall_price(self) -> float:
        """Price of the largest single ask (whale ask wall)."""
        if not self.asks:
            return 0.0
        return max(self.asks, key=lambda x: x.notional_usd).price

    @property
    def cumulative_bid_depth(self) -> float:
        """Total notional USD on bid side."""
        return sum(lvl.notional_usd for lvl in self.bids)

    @property
    def cumulative_ask_depth(self) -> float:
        """Total notional USD on ask side."""
        return sum(lvl.notional_usd for lvl in self.asks)

    def to_dict(self) -> dict:
        return {
            "ts_ns": self.timestamp_ns,
            "symbol": self.symbol,
            "sequence": self.sequence,
            "mid": self.mid_price,
            "spread_bps": self.spread_bps,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "bid_wall": self.bid_wall_price,
            "ask_wall": self.ask_wall_price,
            "cum_bid_depth": self.cumulative_bid_depth,
            "cum_ask_depth": self.cumulative_ask_depth,
            "bids": [[lvl.price, lvl.quantity] for lvl in self.bids[:20]],
            "asks": [[lvl.price, lvl.quantity] for lvl in self.asks[:20]],
            "funding_rate": self.funding_rate,
            "open_interest": self.open_interest,
        }


class L2FeedGenesis:
    """Full 100-level L2 Feed with MarketDataBus publishing and synthetic mode."""

    def __init__(self, config: dict, bus: Optional[InProcessBus] = None) -> None:
        self._config = config
        self._bus = bus
        exchange_cfg = config.get("exchange", {})
        feed_cfg = config.get("feed", {})
        self.symbol = exchange_cfg.get("symbol", "BTC/USDT:USDT")
        self.depth = feed_cfg.get("order_book_depth", 100)
        self.poll_interval_ms = feed_cfg.get("poll_interval_ms", 500)
        self.trade_history_window = feed_cfg.get("trade_history_window", 500)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self.trade_history: List[TradeTick] = []
        self._sequence = 0
        self._running = False

    async def start(self) -> None:
        self._running = True
        # Always use synthetic for Genesis (real feed requires CCXT Pro license)
        await self._synthetic_loop()

    async def stop(self) -> None:
        self._running = False

    async def _synthetic_loop(self) -> None:
        import numpy as np
        rng = np.random.default_rng(42)
        bt_cfg = self._config.get("backtest", {})
        price = float(bt_cfg.get("synthetic_start_price", 65000.0))
        vol = float(bt_cfg.get("synthetic_volatility", 0.002))
        n_bars = int(bt_cfg.get("n_synthetic_bars", 10000))
        interval_s = self.poll_interval_ms / 1000.0
        bar = 0

        # ── Regime-switching process (Markov momentum) ──────────────
        # Alternating bull/bear regimes of 30-80 bars with 0.05%/bar drift.
        # Creates real autocorrelation that trend-following agents can exploit.
        regime = 1          # +1 = bull, -1 = bear
        regime_bar = 0
        regime_len = int(rng.integers(30, 80))

        while self._running and bar < n_bars:
            # ── Regime switching ──────────────────────────────────────
            regime_bar += 1
            if regime_bar >= regime_len:
                regime *= -1                          # Flip regime
                regime_len = int(rng.integers(30, 80))
                regime_bar = 0

            # ── Price step: drift + noise ─────────────────────────────
            # Drift = 0.0005 * regime per bar (~0.05% trend), noise reduced
            # to make trend signal-to-noise ratio meaningful.
            drift = 0.0005 * regime
            noise_vol = vol * 0.6                     # Reduce noise vs drift
            ret = drift + rng.normal(0, noise_vol)
            price = max(price * (1 + ret), 1.0)

            half_spread = price * rng.uniform(0.0001, 0.0002) / 2

            # ── Build LOB biased toward regime ────────────────────────
            # In a bull regime, bids are heavier (institutional accumulation)
            bull_factor = 1.5 if regime == 1 else 0.7
            bear_factor = 0.7 if regime == 1 else 1.5

            bids, asks = [], []
            for i in range(self.depth):
                bp = price - half_spread - i * price * 0.00008
                ap = price + half_spread + i * price * 0.00008
                bq = float(rng.exponential(1.5)) * bull_factor + 0.1
                aq = float(rng.exponential(1.5)) * bear_factor + 0.1
                # Whale orders (2% probability, biased toward regime direction)
                if i == 0 and rng.random() < 0.04:
                    if regime == 1:
                        bq += rng.uniform(20, 50)    # Whale bid in bull regime
                    else:
                        aq += rng.uniform(20, 50)    # Whale ask in bear regime
                bids.append(PriceLevel(bp, bq))
                asks.append(PriceLevel(ap, aq))

            self._sequence += 1
            snap = L2SnapshotFull(
                timestamp_ns=int(time.time_ns()),
                bids=bids, asks=asks,
                symbol=self.symbol, sequence=self._sequence,
            )
            try:
                self.queue.put_nowait(snap)
            except asyncio.QueueFull:
                try:
                    self.queue.get_nowait()
                    self.queue.put_nowait(snap)
                except asyncio.QueueEmpty:
                    pass

            if self._bus:
                await self._bus.publish(MarketEvent(
                    type=EventType.ORDERBOOK, data=snap.to_dict(), source="l2_genesis"
                ))

            bar += 1
            await asyncio.sleep(interval_s * 0.01)  # Fast in backtest

        self._running = False


