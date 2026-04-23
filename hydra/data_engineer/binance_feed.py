"""
Binance Paper Trading Feed
===========================
Connects to Binance public WebSocket bookTicker stream for real BTC/USDT prices,
then synthesizes a 100-level L2 orderbook from the real mid price for Genesis pipeline.

No API key required — uses public Binance streams only.
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
import numpy as np

from hydra.data_engineer.l2_feed_genesis import (
    L2FeedGenesis, PriceLevel, L2SnapshotFull
)
try:
    from hydra.data_engineer.market_bus import MarketEvent, EventType
except ImportError:
    MarketEvent = EventType = None  # type: ignore

logger = logging.getLogger(__name__)


class BinancePaperFeed(L2FeedGenesis):
    """
    Real Binance WebSocket feed for paper trading.

    Stream: wss://stream.binance.com:9443/ws/<symbol>@bookTicker
    Produces real bid/ask mid price + synthetic 100-level LOB depth.
    Falls back to synthetic GBM if Binance is unreachable.
    """

    _STREAM = "wss://stream.binance.com:9443/ws/{symbol}@bookTicker"

    def __init__(self, config: dict, bus=None) -> None:
        super().__init__(config, bus=bus)
        sym = config.get("feed", {}).get("symbol", "BTCUSDT").upper()
        self._ws_url = self._STREAM.format(symbol=sym.lower())
        self._rng = np.random.default_rng()          # fresh seed each run
        self._last_bid = 65000.0
        self._last_ask = 65010.0
        logger.info("BinancePaperFeed | symbol=%s | url=%s", sym, self._ws_url)

    # ── Override start to use real data ──────────────────────────────

    async def start(self) -> None:
        self._running = True
        try:
            import websockets as _ws
        except ImportError:
            logger.warning("websockets not installed — falling back to synthetic feed.")
            await self._synthetic_loop()
            return
        await self._binance_loop(_ws)

    # ── Main loop ────────────────────────────────────────────────────

    async def _binance_loop(self, ws_lib) -> None:
        reconnect_delay = 2.0
        while self._running:
            try:
                async with ws_lib.connect(
                    self._ws_url, ping_interval=20, ping_timeout=10
                ) as ws:
                    reconnect_delay = 2.0
                    logger.info("BinancePaperFeed: connected to Binance stream")
                    async for raw in ws:
                        if not self._running:
                            return
                        try:
                            d = json.loads(raw)
                            bid = float(d["b"])
                            ask = float(d["a"])
                        except (KeyError, ValueError):
                            continue
                        if bid <= 0 or ask <= 0:
                            continue
                        self._last_bid, self._last_ask = bid, ask
                        snap = self._build_snapshot(bid, ask)
                        await self._push(snap)
            except Exception as e:
                if not self._running:
                    return
                logger.warning("BinancePaperFeed reconnecting in %.0fs (%s)", reconnect_delay, e)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30)

    # ── Build synthetic LOB around real mid ──────────────────────────

    def _build_snapshot(self, bid: float, ask: float) -> "L2SnapshotFull":
        mid = (bid + ask) / 2
        spread = ask - bid
        rng = self._rng
        self._sequence += 1

        bids, asks = [], []
        for i in range(self.depth):
            # Level spacing = real spread + exponential decay away from top
            bp = bid  - i * spread * 0.5 * (1 + rng.exponential(0.3))
            ap = ask  + i * spread * 0.5 * (1 + rng.exponential(0.3))
            bq = float(rng.exponential(1.5)) + 0.05
            aq = float(rng.exponential(1.5)) + 0.05
            # Occasional whale walls
            if i == 0 and rng.random() < 0.03:
                bq += rng.uniform(15, 40)
            if i == 2 and rng.random() < 0.02:
                aq += rng.uniform(15, 40)
            bids.append(PriceLevel(bp, bq))
            asks.append(PriceLevel(ap, aq))

        return L2SnapshotFull(
            timestamp_ns=time.time_ns(),
            bids=bids, asks=asks,
            symbol=self.symbol, sequence=self._sequence,
        )

    async def _push(self, snap: "L2SnapshotFull") -> None:
        try:
            self.queue.put_nowait(snap)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(snap)
            except asyncio.QueueEmpty:
                pass
        if self._bus and MarketEvent and EventType:
            await self._bus.publish(MarketEvent(
                type=EventType.ORDERBOOK, data=snap.to_dict(), source="binance_paper"
            ))
