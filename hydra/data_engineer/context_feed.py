"""
Context Feed — Funding Rates / Open Interest / Liquidations
=============================================================
Ingests market-wide sentiment & positioning data via CCXT REST.
Publishes to MarketDataBus as EventType.CONTEXT.

Data Points:
  - Funding Rate (perpetual futures): sentiment proxy
  - Open Interest: market conviction measure
  - Estimated Liquidations: inferred from OI delta + price move

Interpretation:
  High funding + high OI + rising price → crowded long, fade risk
  Negative funding + falling OI        → potential capitulation bottom
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ContextSnapshot:
    """Combined market context at a point in time."""
    timestamp_ns: int
    funding_rate: float       # Current hourly funding rate (fraction)
    funding_rate_annualized: float
    open_interest: float      # OI in USD
    oi_delta: float           # Change in OI since last snapshot
    estimated_liq: float      # Estimated liquidations in USD (abs)
    sentiment: str            # 'BULLISH' | 'BEARISH' | 'NEUTRAL' | 'EXTREME_LONG' | 'EXTREME_SHORT'

    @property
    def is_crowded_long(self) -> bool:
        return self.funding_rate > 0.0005 and self.oi_delta > 0

    @property
    def is_crowded_short(self) -> bool:
        return self.funding_rate < -0.0003 and self.oi_delta > 0

    def to_dict(self) -> dict:
        return {
            "ts_ns": self.timestamp_ns,
            "funding_rate": self.funding_rate,
            "funding_annualized": self.funding_rate_annualized,
            "open_interest": self.open_interest,
            "oi_delta": self.oi_delta,
            "est_liq": self.estimated_liq,
            "sentiment": self.sentiment,
            "crowded_long": self.is_crowded_long,
            "crowded_short": self.is_crowded_short,
        }


class ContextFeed:
    """
    Asynchronous context feed for funding rate, OI, and liquidation data.

    In backtest mode, generates synthetic context data.
    In live mode, polls Binance futures endpoints via CCXT.

    Args:
        config: Hydra YAML config dict.
        bus:    MarketDataBus instance for publishing.
        mode:   'backtest' | 'paper' | 'live'
    """

    def __init__(self, config: dict, bus=None, mode: str = "backtest") -> None:
        self._bus = bus
        self._mode = mode
        self._exchange_name = config.get("exchange", {}).get("name", "binance")
        self._symbol = config.get("exchange", {}).get("symbol", "BTC/USDT:USDT")
        self._poll_interval = config.get("context", {}).get("poll_interval_s", 30)
        self._running = False
        self._last_oi: float = 0.0
        self._last_snapshot: Optional[ContextSnapshot] = None

        import random
        self._rng = random.Random(42)

    async def start(self) -> None:
        self._running = True
        if self._mode == "backtest":
            await self._synthetic_loop()
        else:
            await self._live_loop()

    async def stop(self) -> None:
        self._running = False

    async def _live_loop(self) -> None:
        try:
            import ccxt.async_support as ccxt_async  # type: ignore
            exchange = getattr(ccxt_async, self._exchange_name)(
                {"defaultType": "future", "enableRateLimit": True}
            )
            await exchange.load_markets()
            while self._running:
                try:
                    snap = await self._fetch_context(exchange)
                    self._last_snapshot = snap
                    if self._bus:
                        from hydra.data_engineer.market_bus import MarketEvent, EventType
                        await self._bus.publish(MarketEvent(
                            type=EventType.CONTEXT,
                            data=snap.to_dict(),
                            source="context_feed",
                        ))
                except Exception as e:
                    logger.warning("ContextFeed error: %s", e)
                await asyncio.sleep(self._poll_interval)
            await exchange.close()
        except Exception as e:
            logger.warning("ContextFeed live mode failed: %s — falling back to synthetic", e)
            await self._synthetic_loop()

    async def _fetch_context(self, exchange) -> ContextSnapshot:
        ts_ns = int(time.time_ns())
        funding_rate = 0.0001  # default fallback
        oi = self._last_oi or 1e9

        try:
            fr_data = await exchange.fetch_funding_rate(self._symbol)
            funding_rate = float(fr_data.get("fundingRate", 0.0001))
        except Exception:
            pass

        try:
            oi_data = await exchange.fetch_open_interest(self._symbol)
            oi = float(oi_data.get("openInterestValue", oi_data.get("openInterest", self._last_oi)))
        except Exception:
            pass

        oi_delta = oi - self._last_oi
        self._last_oi = oi

        # Estimate liquidations from OI drop + price sensitivity
        est_liq = abs(oi_delta) * 0.1 if oi_delta < 0 else 0.0

        sentiment = self._classify_sentiment(funding_rate, oi_delta)
        return ContextSnapshot(
            timestamp_ns=ts_ns,
            funding_rate=funding_rate,
            funding_rate_annualized=funding_rate * 3 * 365,
            open_interest=oi,
            oi_delta=oi_delta,
            estimated_liq=est_liq,
            sentiment=sentiment,
        )

    async def _synthetic_loop(self) -> None:
        """Generate synthetic context for backtesting."""
        rng = self._rng
        oi = 8_000_000_000.0  # $8B base OI

        while self._running:
            funding_rate = rng.gauss(0.0001, 0.0003)
            oi_change = rng.gauss(0, oi * 0.001)
            oi = max(1e8, oi + oi_change)
            est_liq = max(0.0, -oi_change * 0.05) if oi_change < 0 else 0.0

            snap = ContextSnapshot(
                timestamp_ns=int(time.time_ns()),
                funding_rate=funding_rate,
                funding_rate_annualized=funding_rate * 3 * 365,
                open_interest=oi,
                oi_delta=oi_change,
                estimated_liq=est_liq,
                sentiment=self._classify_sentiment(funding_rate, oi_change),
            )
            self._last_snapshot = snap

            if self._bus:
                from hydra.data_engineer.market_bus import MarketEvent, EventType
                await self._bus.publish(MarketEvent(
                    type=EventType.CONTEXT, data=snap.to_dict(), source="context_feed"
                ))

            await asyncio.sleep(0.5)  # Fast in backtest

        self._running = False

    @staticmethod
    def _classify_sentiment(funding_rate: float, oi_delta: float) -> str:
        if funding_rate > 0.001:
            return "EXTREME_LONG"
        elif funding_rate > 0.0003:
            return "BULLISH"
        elif funding_rate < -0.001:
            return "EXTREME_SHORT"
        elif funding_rate < -0.0003:
            return "BEARISH"
        return "NEUTRAL"

    @property
    def last_snapshot(self) -> Optional[ContextSnapshot]:
        return self._last_snapshot
