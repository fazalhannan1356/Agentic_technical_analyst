"""
Liquidity Heatmap — Rolling Volume-at-Price (VAP)
===================================================
Builds a dynamic Volume-at-Price heatmap from order book snapshots to
identify Whale Cluster Zones — price levels with anomalously high liquidity.

Algorithm:
  1. Each snapshot contributes bid/ask quantities to a price-bucketed histogram.
  2. Buckets are price-rounded to `tick_size` resolution.
  3. The rolling window exponentially decays older contributions.
  4. Whale Cluster Zones are levels where total notional > μ + N×σ.

Published to MarketDataBus as EventType.HEATMAP.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from hydra.data_engineer.market_bus import MarketDataBus, MarketEvent, EventType, InProcessBus

logger = logging.getLogger(__name__)


@dataclass
class WhaleClusters:
    """Price levels with anomalously large accumulated volume."""
    support_levels: List[float]   # bid-side whale clusters
    resistance_levels: List[float]  # ask-side whale clusters
    heatmap: Dict[float, float]   # full price → notional map
    mid_price: float
    timestamp_ns: int

    def nearest_support(self, price: float) -> Optional[float]:
        candidates = [s for s in self.support_levels if s < price]
        return max(candidates) if candidates else None

    def nearest_resistance(self, price: float) -> Optional[float]:
        candidates = [r for r in self.resistance_levels if r > price]
        return min(candidates) if candidates else None

    def support_strength(self, price: float, window_pct: float = 0.005) -> float:
        """Returns normalised support strength near price."""
        lo, hi = price * (1 - window_pct), price * (1 + window_pct)
        zone = {k: v for k, v in self.heatmap.items() if lo <= k <= hi}
        return sum(zone.values()) / (max(self.heatmap.values()) + 1e-8) if zone else 0.0


class LiquidityHeatmap:
    """
    Rolling Volume-at-Price heatmap.

    Args:
        config:       Hydra YAML config dict.
        bus:          MarketDataBus for publishing updates (optional).
        tick_size:    Price bucket granularity (default 1.0 for BTC in USD).
        decay:        Exponential decay factor per bar (0.995 = slow decay).
        sigma_thresh: Whale cluster zone threshold (μ + N×σ).
    """

    def __init__(
        self,
        config: dict,
        bus: Optional[InProcessBus] = None,
        tick_size: float = 10.0,
        decay: float = 0.995,
        sigma_thresh: float = 2.5,
    ) -> None:
        self._bus = bus
        self._tick = tick_size
        self._decay = decay
        self._sigma = sigma_thresh
        self._heatmap: Dict[float, float] = defaultdict(float)
        self._bar_count = 0
        self._last_clusters: Optional[WhaleClusters] = None

    def _bucket(self, price: float) -> float:
        return round(price / self._tick) * self._tick

    def update(self, bids: list, asks: list, mid_price: float, ts_ns: int) -> WhaleClusters:
        """
        Ingest one L2 snapshot side and update the heatmap.

        Args:
            bids:      List of (price, qty) tuples.
            asks:      List of (price, qty) tuples.
            mid_price: Current mid price.
            ts_ns:     Snapshot timestamp in ns.

        Returns:
            WhaleClusters with identified support/resistance zones.
        """
        self._bar_count += 1

        # Exponential decay
        for k in list(self._heatmap.keys()):
            self._heatmap[k] *= self._decay
            if self._heatmap[k] < 1.0:
                del self._heatmap[k]

        # Accumulate bid / ask notionals
        for price, qty in bids:
            bucket = self._bucket(price)
            self._heatmap[bucket] += price * qty  # USD notional

        for price, qty in asks:
            bucket = self._bucket(price)
            self._heatmap[bucket] += price * qty

        # Detect whale clusters (μ + N×σ threshold)
        values = np.array(list(self._heatmap.values()), dtype=np.float64)
        if len(values) < 5:
            clusters = WhaleClusters([], [], dict(self._heatmap), mid_price, ts_ns)
            self._last_clusters = clusters
            return clusters

        mu, sigma = values.mean(), values.std()
        threshold = mu + self._sigma * sigma

        support_lvls, resist_lvls = [], []
        for price_lvl, notional in self._heatmap.items():
            if notional >= threshold:
                if price_lvl < mid_price:
                    support_lvls.append(price_lvl)
                else:
                    resist_lvls.append(price_lvl)

        support_lvls.sort(reverse=True)   # strongest first
        resist_lvls.sort()

        clusters = WhaleClusters(
            support_levels=support_lvls,
            resistance_levels=resist_lvls,
            heatmap=dict(self._heatmap),
            mid_price=mid_price,
            timestamp_ns=ts_ns,
        )
        self._last_clusters = clusters
        return clusters

    async def publish(self, clusters: WhaleClusters) -> None:
        if self._bus:
            data = {
                "mid": clusters.mid_price,
                "ts_ns": clusters.timestamp_ns,
                "support": clusters.support_levels[:5],
                "resistance": clusters.resistance_levels[:5],
                "top_levels": sorted(
                    clusters.heatmap.items(), key=lambda x: x[1], reverse=True
                )[:20],
            }
            await self._bus.publish(MarketEvent(
                type=EventType.HEATMAP, data=data, source="heatmap"
            ))

    @property
    def last_clusters(self) -> Optional[WhaleClusters]:
        return self._last_clusters

    def top_levels(self, n: int = 10) -> List[Tuple[float, float]]:
        """Return top-N price levels by notional (price, notional_usd)."""
        return sorted(self._heatmap.items(), key=lambda x: x[1], reverse=True)[:n]
