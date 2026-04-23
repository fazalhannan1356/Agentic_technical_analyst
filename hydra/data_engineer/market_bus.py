"""
Unified Market Data Bus — Redis Pub/Sub
========================================
Central nervous system of the Hydra Genesis engine.
All data producers publish to named channels; all agents subscribe.

Channels (EventType):
  ORDERBOOK   — Full 100-level L2 snapshot (JSON)
  HEATMAP     — Volume-at-Price cluster update
  CONTEXT     — Funding rate / OI / Liquidations update
  TRADE       — Individual trade tick
  SIGNAL      — Specialist agent signal (JSON TradeStruct)

Usage (publisher):
    bus = MarketDataBus.from_config(config)
    await bus.connect()
    await bus.publish(MarketEvent(type=EventType.ORDERBOOK, data=snapshot_dict))

Usage (subscriber):
    async for event in bus.subscribe(EventType.ORDERBOOK, EventType.HEATMAP):
        handle(event)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Event Types
# ─────────────────────────────────────────────

class EventType(str, Enum):
    ORDERBOOK = "hydra:orderbook"
    HEATMAP   = "hydra:heatmap"
    CONTEXT   = "hydra:context"
    TRADE     = "hydra:trade"
    SIGNAL    = "hydra:signal"
    HEARTBEAT = "hydra:heartbeat"


# ─────────────────────────────────────────────
#  Market Event
# ─────────────────────────────────────────────

@dataclass
class MarketEvent:
    """
    A single typed event published to the bus.

    Attributes:
        type:        EventType channel name.
        data:        Serializable payload (dict / dataclass / primitive).
        timestamp_ns: Creation time in nanoseconds (auto-filled if 0).
        source:      Identifying label of the producer ('l2_feed', 'context_feed', etc.).
    """
    type: EventType
    data: Any
    timestamp_ns: int = field(default_factory=time.time_ns)
    source: str = "unknown"

    def to_json(self) -> str:
        payload = {
            "type": self.type.value,
            "ts_ns": self.timestamp_ns,
            "source": self.source,
            "data": self.data,
        }
        return json.dumps(payload, default=str)

    @classmethod
    def from_json(cls, raw: str) -> "MarketEvent":
        d = json.loads(raw)
        return cls(
            type=EventType(d["type"]),
            data=d["data"],
            timestamp_ns=d.get("ts_ns", 0),
            source=d.get("source", "unknown"),
        )


# ─────────────────────────────────────────────
#  In-Process Bus (fallback — no Redis required)
# ─────────────────────────────────────────────

class InProcessBus:
    """
    Pure-asyncio implementation of the Market Data Bus.
    Used when Redis is unavailable or in test/backtest mode.
    Implements the same interface as RedisBus.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[asyncio.Queue]] = {}
        self._running = False
        self._stats: Dict[str, int] = {}

    async def connect(self) -> None:
        self._running = True
        logger.info("MarketDataBus: In-process mode (no Redis dependency).")

    async def disconnect(self) -> None:
        self._running = False

    async def publish(self, event: MarketEvent) -> None:
        """Distribute event to all subscribers of this channel."""
        queues = self._subscribers.get(event.type, [])
        self._stats[event.type.value] = self._stats.get(event.type.value, 0) + 1
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer — drop oldest
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except asyncio.QueueEmpty:
                    pass

    def create_subscription(
        self,
        *channels: EventType,
        maxsize: int = 2000,
    ) -> "BusSubscription":
        """
        Create a subscription handle. The caller iterates it as an async generator.

        Returns:
            BusSubscription context manager that auto-unsubscribes on exit.
        """
        q: asyncio.Queue[MarketEvent] = asyncio.Queue(maxsize=maxsize)
        for ch in channels:
            self._subscribers.setdefault(ch, []).append(q)
        return BusSubscription(self, q, set(channels))

    def unsubscribe(self, q: asyncio.Queue, channels: Set[EventType]) -> None:
        for ch in channels:
            lst = self._subscribers.get(ch, [])
            try:
                lst.remove(q)
            except ValueError:
                pass

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


class BusSubscription:
    """Context manager + async generator for bus subscriptions."""

    def __init__(
        self,
        bus: InProcessBus,
        queue: asyncio.Queue,
        channels: Set[EventType],
    ) -> None:
        self._bus = bus
        self._queue = queue
        self._channels = channels

    async def __aenter__(self) -> "BusSubscription":
        return self

    async def __aexit__(self, *_) -> None:
        self._bus.unsubscribe(self._queue, self._channels)

    def __aiter__(self) -> "BusSubscription":
        return self

    async def __anext__(self) -> MarketEvent:
        return await self._queue.get()

    async def get(self, timeout: float = 10.0) -> Optional[MarketEvent]:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


# ─────────────────────────────────────────────
#  Redis Bus (optional — requires redis-py)
# ─────────────────────────────────────────────

class RedisBus:
    """
    Redis Pub/Sub implementation.
    Requires: pip install redis[hiredis]
    Falls back gracefully to InProcessBus if redis is unavailable.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0) -> None:
        self._host = host
        self._port = port
        self._db = db
        self._redis = None
        self._pubsub = None
        self._running = False

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis  # type: ignore
            self._redis = aioredis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                decode_responses=True,
            )
            await self._redis.ping()
            self._pubsub = self._redis.pubsub()
            self._running = True
            logger.info(
                "MarketDataBus: Connected to Redis at %s:%d", self._host, self._port
            )
        except Exception as e:
            logger.warning("Redis unavailable (%s) — falling back to in-process bus.", e)
            raise

    async def disconnect(self) -> None:
        self._running = False
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.aclose()

    async def publish(self, event: MarketEvent) -> None:
        if self._redis:
            await self._redis.publish(event.type.value, event.to_json())

    def create_subscription(
        self, *channels: EventType, maxsize: int = 2000
    ) -> "RedisBusSubscription":
        return RedisBusSubscription(self._pubsub, channels, maxsize)


class RedisBusSubscription:
    def __init__(self, pubsub, channels, maxsize: int) -> None:
        self._pubsub = pubsub
        self._channels = channels
        self._maxsize = maxsize

    async def __aenter__(self) -> "RedisBusSubscription":
        await self._pubsub.subscribe(*[ch.value for ch in self._channels])
        return self

    async def __aexit__(self, *_) -> None:
        await self._pubsub.unsubscribe(*[ch.value for ch in self._channels])

    def __aiter__(self) -> "RedisBusSubscription":
        return self

    async def __anext__(self) -> MarketEvent:
        async for msg in self._pubsub.listen():
            if msg and msg["type"] == "message":
                return MarketEvent.from_json(msg["data"])
        raise StopAsyncIteration

    async def get(self, timeout: float = 10.0) -> Optional[MarketEvent]:
        try:
            msg = await asyncio.wait_for(self._pubsub.get_message(timeout=timeout), timeout=timeout + 1)
            if msg and msg["type"] == "message":
                return MarketEvent.from_json(msg["data"])
        except asyncio.TimeoutError:
            pass
        return None


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

class MarketDataBus:
    """
    Factory that returns either a Redis bus or in-process fallback.

    Usage:
        bus = MarketDataBus.from_config(config)
        await bus.connect()
        sub = bus.create_subscription(EventType.ORDERBOOK, EventType.HEATMAP)
        async with sub:
            async for event in sub:
                ...
    """

    @staticmethod
    def from_config(config: dict) -> "InProcessBus | RedisBus":
        bus_cfg = config.get("bus", {})
        use_redis = bus_cfg.get("redis_enabled", False)
        if use_redis:
            host = bus_cfg.get("redis_host", "localhost")
            port = int(bus_cfg.get("redis_port", 6379))
            bus = RedisBus(host=host, port=port)
            return bus  # type: ignore[return-value]
        return InProcessBus()
