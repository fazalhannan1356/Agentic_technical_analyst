"""
Fractional Differentiation — Stationarity-preserving transformation.
Reference: López de Prado (2018) AFML Chapter 5.

Finds minimum d that achieves ADF stationarity while retaining long-term memory.
"""
from __future__ import annotations
import logging
from collections import deque
from typing import Optional, Sequence, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def _ffd_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Compute Fixed-width window fractional differencing weights."""
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1], dtype=np.float64)


class FractionalDifferentiator:
    """
    Streaming-capable FFD differentiator.

    Args:
        d: Differencing order (0, 1]. Typical values: 0.3–0.5.
        threshold: Weight truncation threshold for FFD.

    Example:
        fd = FractionalDifferentiator(d=0.4)
        for price in series:
            val = fd.update(price)  # None during warm-up
    """
    def __init__(self, d: float = 0.4, threshold: float = 1e-5) -> None:
        if not 0 < d <= 1:
            raise ValueError(f"d must be in (0, 1], got {d}")
        self.d = d
        self._weights = _ffd_weights(d, threshold)
        self._window = len(self._weights)
        self._buffer: deque = deque(maxlen=self._window)
        logger.info("FractionalDifferentiator | d=%.3f | window=%d", d, self._window)

    def update(self, price: float) -> Optional[float]:
        """Process one price. Returns None during warm-up."""
        self._buffer.append(price)
        if len(self._buffer) < self._window:
            return None
        return float(np.dot(self._weights, np.array(self._buffer, dtype=np.float64)))

    def transform(self, prices: Sequence[float]) -> np.ndarray:
        """Batch transform. First (window-1) entries are NaN."""
        arr = np.array(prices, dtype=np.float64)
        n = len(arr)
        result = np.full(n, np.nan)
        for i in range(self._window - 1, n):
            result[i] = float(np.dot(self._weights, arr[i - self._window + 1: i + 1]))
        return result

    def reset(self) -> None:
        self._buffer.clear()

    @staticmethod
    def find_min_d(
        prices: Sequence[float],
        d_range: Tuple[float, float] = (0.05, 1.0),
        step: float = 0.05,
        adf_threshold: float = -3.5,
    ) -> float:
        """Grid-search for minimum d achieving ADF stationarity."""
        try:
            from statsmodels.tsa.stattools import adfuller  # type: ignore
        except ImportError:
            logger.warning("statsmodels not installed — using default d=0.4")
            return 0.4
        arr = np.array(prices, dtype=np.float64)
        for d in np.arange(d_range[0], d_range[1] + step, step):
            fd = FractionalDifferentiator(d=float(d))
            series = fd.transform(arr)
            series = series[~np.isnan(series)]
            if len(series) < 20:
                continue
            try:
                adf_stat, *_ = adfuller(series, maxlag=1, regression="c", autolag=None)
                if adf_stat < adf_threshold:
                    logger.info("MinD found: d=%.2f (ADF=%.3f)", d, adf_stat)
                    return float(d)
            except Exception:
                continue
        return float(d_range[1])

    @property
    def warmup_bars(self) -> int:
        return self._window

    @property
    def buffer_fill_pct(self) -> float:
        return min(len(self._buffer) / self._window, 1.0)
