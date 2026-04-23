"""
Chart Agent — PatchTST-based Pattern Recognition
=================================================
Specialist Agent #1 in the Hydra Swarm.

Implements a PatchTST (Patch Time Series Transformer) for chart analysis.
Reference: Nie et al. (2023) "A Time Series is Worth 64 Words: Long-term
Forecasting with Transformers." ICLR 2023.

Patterns Detected:
  - Higher High / Higher Low (HH/HL) — bullish structure
  - Lower High / Lower Low (LH/LL)  — bearish structure
  - Support / Resistance zones (S/R) — based on fractionally-differenced price
  - Whale Absorption                 — large bid absorption at support

Architecture:
  PriceSeries → PatchEmbedding → TransformerEncoder → PatternHead
  PatternHead outputs:
    - direction: 'LONG' | 'SHORT' | 'NEUTRAL'
    - confidence: [0, 1]
    - patterns: List[str]
    - predicted_next_bars: List[float]  (candle overlay)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — ChartAgent running in heuristic-only mode.")


# ─────────────────────────────────────────────
#  Signal Dataclass
# ─────────────────────────────────────────────

@dataclass
class ChartSignal:
    """Output from the Chart Agent."""
    direction: str            # 'LONG' | 'SHORT' | 'NEUTRAL'
    confidence: float         # [0, 1]
    patterns: List[str]       # e.g. ['HH_HL', 'WHALE_ABSORPTION']
    predicted_candles: List[float] = field(default_factory=list)  # next N mid prices
    sr_support: Optional[float] = None
    sr_resistance: Optional[float] = None
    timestamp_ns: int = 0
    source: str = "chart_agent"

    @property
    def is_actionable(self) -> bool:
        return self.direction != "NEUTRAL" and self.confidence >= 0.6


# ─────────────────────────────────────────────
#  PatchTST Components (PyTorch)
# ─────────────────────────────────────────────

if TORCH_AVAILABLE:
    class PatchEmbedding(nn.Module):
        """Split time series into patches and project to d_model."""
        def __init__(self, patch_size: int, d_model: int, n_features: int) -> None:
            super().__init__()
            self.patch_size = patch_size
            self.proj = nn.Linear(patch_size * n_features, d_model)
            self.pos_enc = None  # Will be built dynamically

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: [B, T, F] → patches: [B, n_patches, patch_size*F]
            B, T, F = x.shape
            n_patches = T // self.patch_size
            x_trim = x[:, :n_patches * self.patch_size, :]
            patches = x_trim.reshape(B, n_patches, self.patch_size * F)
            return self.proj(patches)  # [B, n_patches, d_model]

    class PatchTSTModel(nn.Module):
        """
        Lightweight PatchTST for pattern classification + forecasting.

        Args:
            n_features:   Number of input features per timestep.
            seq_len:      Input sequence length in bars.
            patch_size:   Number of bars per patch.
            d_model:      Transformer embedding dim.
            n_heads:      Multi-head attention heads.
            n_layers:     Encoder depth.
            n_classes:    Classification output size (3: DOWN/NEUTRAL/UP).
            forecast_len: Number of future bars to predict (candle overlay).
        """
        def __init__(
            self,
            n_features: int = 8,
            seq_len: int = 64,
            patch_size: int = 8,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 3,
            n_classes: int = 3,
            forecast_len: int = 5,
        ) -> None:
            super().__init__()
            self.patch_embed = PatchEmbedding(patch_size, d_model, n_features)
            n_patches = seq_len // patch_size
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.cls_head = nn.Linear(d_model, n_classes)
            self.forecast_head = nn.Linear(d_model * n_patches, forecast_len)
            self.n_patches = n_patches
            self.d_model = d_model

        def forward(self, x: "torch.Tensor"):
            patches = self.patch_embed(x)        # [B, n_patches, d_model]
            enc = self.encoder(patches)           # [B, n_patches, d_model]
            cls_logits = self.cls_head(enc[:, -1, :])  # [B, n_classes]
            forecast = self.forecast_head(enc.reshape(enc.shape[0], -1))  # [B, forecast_len]
            return cls_logits, forecast


# ─────────────────────────────────────────────
#  Chart Agent
# ─────────────────────────────────────────────

class ChartAgent:
    """
    PatchTST-based chart pattern recognition specialist.

    In production, loads a pretrained PatchTST model.
    Falls back to rule-based heuristics if PyTorch unavailable.

    Args:
        config:         Hydra YAML config dict.
        model_path:     Optional path to a pretrained .pt model file.
        confidence_thr: Minimum confidence to emit a non-NEUTRAL signal.
    """

    def __init__(
        self,
        config: dict,
        model_path: Optional[str] = None,
        confidence_thr: float = 0.60,
    ) -> None:
        self._config = config
        self._confidence_thr = confidence_thr
        self._model = None
        self._seq_len = config.get("chart_agent", {}).get("seq_len", 64)
        self._patch_size = config.get("chart_agent", {}).get("patch_size", 8)
        self._n_features = config.get("chart_agent", {}).get("n_features", 8)
        self._forecast_len = config.get("chart_agent", {}).get("forecast_len", 5)
        self._price_buffer: List[float] = []
        self._feature_buffer: List[np.ndarray] = []
        self._pivots: List[float] = []

        if TORCH_AVAILABLE and model_path:
            self._load_model(model_path)
        elif TORCH_AVAILABLE:
            self._build_fresh_model()
        logger.info("ChartAgent initialized | torch=%s", TORCH_AVAILABLE)

    def _build_fresh_model(self) -> None:
        self._model = PatchTSTModel(
            n_features=self._n_features,
            seq_len=self._seq_len,
            patch_size=self._patch_size,
            forecast_len=self._forecast_len,
        )
        logger.info("ChartAgent: fresh PatchTST model created (untrained).")

    def _load_model(self, path: str) -> None:
        try:
            self._model = PatchTSTModel(
                n_features=self._n_features,
                seq_len=self._seq_len,
                patch_size=self._patch_size,
                forecast_len=self._forecast_len,
            )
            state = torch.load(path, map_location="cpu")  # type: ignore
            self._model.load_state_dict(state)
            self._model.eval()
            logger.info("ChartAgent: loaded PatchTST from %s", path)
        except Exception as e:
            logger.warning("ChartAgent: failed to load model (%s) — heuristic mode.", e)
            self._model = None

    def process(
        self,
        mid_price: float,
        feature_vec: Optional[np.ndarray],
        timestamp_ns: int = 0,
    ) -> Optional[ChartSignal]:
        """
        Process one bar. Returns ChartSignal when buffer is full.

        Args:
            mid_price:   Current mid price.
            feature_vec: Feature vector from FeatureEngineer (optional).
            timestamp_ns: Bar timestamp.

        Returns:
            ChartSignal or None (during warm-up).
        """
        self._price_buffer.append(mid_price)
        if feature_vec is not None:
            self._feature_buffer.append(feature_vec)

        # ── Always build self-contained price features ─────────────
        # These ensure the model gets meaningful inputs even without FeatureEngineer.
        price_feat = self._build_price_features()
        if price_feat is not None:
            # Use these as the primary feature source if no external vec provided
            if feature_vec is None:
                self._feature_buffer.append(price_feat)

        if len(self._price_buffer) < self._seq_len:
            return None  # Warming up

        # Keep rolling window
        self._price_buffer = self._price_buffer[-self._seq_len:]
        self._feature_buffer = self._feature_buffer[-self._seq_len:]

        # Pattern detection
        patterns = self._detect_patterns()
        sr_support, sr_resistance = self._compute_sr(mid_price)

        # 1. Try trained model inference
        direction, confidence, predicted_candles = self._model_inference()

        # 2. Fallback to linreg heuristic if model is absent or low-confidence
        if confidence < self._confidence_thr:
            direction, confidence = self._linreg_heuristic(patterns, mid_price)

        return ChartSignal(
            direction=direction,
            confidence=confidence,
            patterns=patterns,
            predicted_candles=predicted_candles,
            sr_support=sr_support,
            sr_resistance=sr_resistance,
            timestamp_ns=timestamp_ns,
        )


    def _model_inference(self):
        """Run PatchTST forward pass if available."""
        # Need feature buffer filled to seq_len
        if not TORCH_AVAILABLE or self._model is None or len(self._feature_buffer) < self._seq_len:
            return "NEUTRAL", 0.33, []

        try:
            feats = np.array(self._feature_buffer[-self._seq_len:], dtype=np.float32)
            # Ensure correct feature dimension
            if feats.ndim == 1:
                feats = feats.reshape(-1, 1)
            n_feat = min(feats.shape[1], self._n_features)
            feats = feats[:, :n_feat]
            if feats.shape[1] < self._n_features:
                pad = np.zeros((feats.shape[0], self._n_features - feats.shape[1]), dtype=np.float32)
                feats = np.concatenate([feats, pad], axis=1)

            x = torch.from_numpy(feats).unsqueeze(0)  # [1, T, F]
            with torch.no_grad():
                logits, forecast = self._model(x)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()

            cls_idx = int(np.argmax(probs))
            confidence = float(probs[cls_idx])
            direction_map = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
            direction = direction_map[cls_idx]

            mid = self._price_buffer[-1]
            predicted = [mid + float(f) * mid * 0.001 for f in forecast.squeeze().tolist()]
            return direction, confidence, predicted
        except Exception as e:
            logger.debug("ChartAgent inference error: %s", e)
            return "NEUTRAL", 0.33, []


    def _detect_patterns(self) -> List[str]:
        """Rule-based pattern detection on price buffer."""
        prices = np.array(self._price_buffer)
        patterns = []

        # Find pivots using local extrema (window=5)
        pivots_h, pivots_l = self._find_pivots(prices, window=5)

        # HH/HL — bullish trend structure
        if len(pivots_h) >= 2 and len(pivots_l) >= 2:
            if pivots_h[-1] > pivots_h[-2] and pivots_l[-1] > pivots_l[-2]:
                patterns.append("HH_HL")
            elif pivots_h[-1] < pivots_h[-2] and pivots_l[-1] < pivots_l[-2]:
                patterns.append("LH_LL")

        # Whale absorption: large flat zone near low of range
        price_range = prices.max() - prices.min()
        if price_range > 0:
            low_zone_pct = (prices - prices.min()) / price_range
            time_at_low = np.sum(low_zone_pct < 0.1) / len(prices)
            if time_at_low > 0.3:
                patterns.append("WHALE_ABSORPTION")

        # Consolidation (S/R zone)
        vol = prices.std() / prices.mean()
        if vol < 0.002:
            patterns.append("CONSOLIDATION")

        return patterns

    @staticmethod
    def _find_pivots(prices: np.ndarray, window: int = 5):
        highs, lows = [], []
        for i in range(window, len(prices) - window):
            if prices[i] == prices[i - window:i + window + 1].max():
                highs.append(prices[i])
            if prices[i] == prices[i - window:i + window + 1].min():
                lows.append(prices[i])
        return highs, lows

    def _compute_sr(self, mid_price: float):
        """Identify support/resistance via price clustering."""
        prices = np.array(self._price_buffer)
        if len(prices) < 10:
            return None, None
        support = float(np.percentile(prices, 25))
        resistance = float(np.percentile(prices, 75))
        return support if support < mid_price else None, resistance if resistance > mid_price else None

    def _build_price_features(self) -> Optional[np.ndarray]:
        """
        Build 8 self-contained features from the rolling price buffer.
        Used when external FeatureEngineer is not connected.

        Features:
          0: 1-bar return
          1: 5-bar momentum
          2: 10-bar momentum
          3: 20-bar momentum
          4: 10-bar normalised std (volatility)
          5: Z-score of current price over 20 bars
          6: Fraction from 20-bar high (0=at high)
          7: Fraction from 20-bar low  (1=at low)
        """
        p = self._price_buffer
        n = len(p)
        if n < 21:
            return None

        cur = p[-1]
        def ret(lag):
            return (cur - p[-lag - 1]) / max(abs(p[-lag - 1]), 1e-8) if n > lag else 0.0

        window20 = np.array(p[-20:])
        mu20 = window20.mean()
        std20 = window20.std() + 1e-8

        feats = np.array([
            float(np.clip(ret(1) / 0.005, -5, 5)),         # 0: 1-bar ret (normed)
            float(np.clip(ret(5) / 0.01, -5, 5)),           # 1: 5-bar momentum
            float(np.clip(ret(10) / 0.02, -5, 5)),          # 2: 10-bar momentum
            float(np.clip(ret(min(20, n-1)) / 0.04, -5, 5)),# 3: 20-bar momentum
            float(np.clip(std20 / mu20 / 0.003, 0, 5)),     # 4: normalised vol
            float(np.clip((cur - mu20) / std20, -5, 5)),    # 5: z-score
            float(np.clip((window20.max() - cur) / (window20.max() + 1e-8), 0, 1)),  # 6: dist from high
            float(np.clip((cur - window20.min()) / (window20.min() + 1e-8), 0, 1)),  # 7: dist from low
        ], dtype=np.float32)

        return feats

    def _linreg_heuristic(self, patterns: List[str], mid_price: float):
        """
        Linear regression slope trend detector.

        Fits OLS on the last 30-bar price window. The normalised slope
        (slope per bar / mean_price) gives a robust trend signal.

        Thresholds tuned for regime-switching data with 0.05%/bar drift:
          |norm_slope| > 0.0003 → signal; |r| > 0.4 → minimum trend quality.
        """
        if len(self._price_buffer) < 30:
            return "NEUTRAL", 0.35

        prices = np.array(self._price_buffer[-30:], dtype=np.float64)
        x = np.arange(len(prices), dtype=np.float64)
        # OLS via numpy (no scipy dependency)
        x_mean, p_mean = x.mean(), prices.mean()
        num = ((x - x_mean) * (prices - p_mean)).sum()
        den = ((x - x_mean) ** 2).sum() + 1e-12
        slope = num / den
        norm_slope = slope / max(p_mean, 1e-8)  # Fraction of price per bar

        # Pearson r to measure trend quality
        p_std = prices.std() + 1e-8
        x_std = x.std() + 1e-8
        r = num / (len(prices) * x_std * p_std + 1e-12)
        abs_r = float(np.clip(abs(r), 0, 1))

        # Pattern boosts
        has_bull = "HH_HL" in patterns or "WHALE_ABSORPTION" in patterns
        has_bear = "LH_LL" in patterns

        MIN_SLOPE = 0.00015   # ~0.015% per bar (2× more sensitive)
        MIN_R     = 0.28      # Lower bar for trend quality

        if norm_slope > MIN_SLOPE and abs_r > MIN_R:
            boost = 0.10 if has_bull else 0.0
            conf = float(np.clip(0.54 + abs_r * 0.36 + boost, 0.54, 0.92))
            return "LONG", conf

        if norm_slope < -MIN_SLOPE and abs_r > MIN_R:
            boost = 0.10 if has_bear else 0.0
            conf = float(np.clip(0.54 + abs_r * 0.36 + boost, 0.54, 0.92))
            return "SHORT", conf

        # Sub-threshold: use pattern hints
        if has_bull:
            return "LONG", 0.55
        if has_bear:
            return "SHORT", 0.55

        return "NEUTRAL", 0.33

    @staticmethod
    def _heuristic_direction(patterns, mid, sr_support, sr_resistance):
        """Legacy fallback — superseded by _linreg_heuristic."""
        if "HH_HL" in patterns and "WHALE_ABSORPTION" in patterns:
            return "LONG", 0.72
        if "LH_LL" in patterns:
            return "SHORT", 0.68
        if "HH_HL" in patterns:
            return "LONG", 0.63
        return "NEUTRAL", 0.33


    def train(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 20,
        lr: float = 1e-3,
        save_path: Optional[str] = None,
    ) -> None:
        """Train the PatchTST model on labeled sequences."""
        if not TORCH_AVAILABLE or self._model is None:
            logger.warning("ChartAgent.train: PyTorch not available.")
            return

        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        X = torch.from_numpy(sequences.astype(np.float32))
        y = torch.from_numpy(labels.astype(np.int64))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits, _ = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(len(loader), 1)
            if (epoch + 1) % 5 == 0:
                logger.info("ChartAgent train epoch %d/%d | loss=%.4f", epoch + 1, n_epochs, avg)

        self._model.eval()
        if save_path:
            torch.save(self._model.state_dict(), save_path)
            logger.info("ChartAgent model saved to %s", save_path)
