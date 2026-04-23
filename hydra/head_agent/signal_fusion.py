"""
Head Agent — Weighted Consensus Signal Fusion Engine
=====================================================
The Hydra Master Orchestrator that fuses three specialist signals:
  1. ChartAgent  (PatchTST)  — Technical pattern analysis
  2. RLAgent     (PPO)       — Dynamic leverage + Kelly sizing
  3. LLMAgent    (Claude)    — Whale intent inference

Fusion Algorithm:
  1. Each specialist emits (direction, confidence).
  2. Weighted consensus: W_chart=0.35, W_rl=0.40, W_llm=0.25
  3. Net score = Σ (weight × confidence × direction_sign)
  4. If |score| > threshold → Trade Struct generated.

Veto Rules (any single condition blocks the trade):
  RULE-1: RL says SHORT + Heatmap shows Heavy Support (depth_ratio > 1.5 at support)
  RULE-2: LLM veto_recommendation = True
  RULE-3: Chart confidence < 0.55 AND RL confidence < 0.55
  RULE-4: Funding rate signals EXTREME_LONG for a new LONG trade
  RULE-5: Head consensus score < min_consensus_score threshold

Output: TradeStruct (JSON-serializable) passed to Engine for execution.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional

from hydra.specialist_agents.chart_agent import ChartSignal
from hydra.specialist_agents.rl_agent import RLDecision
from hydra.specialist_agents.llm_agent import LLMSignal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Veto Reason
# ─────────────────────────────────────────────

class VetoReason(str, Enum):
    RL_SHORT_VS_SUPPORT = "RL_SHORT_VS_HEAVY_SUPPORT"
    LLM_VETO           = "LLM_VETO_RECOMMENDATION"
    LOW_CONFIDENCE      = "ALL_AGENTS_LOW_CONFIDENCE"
    CROWDED_LONG        = "EXTREME_LONG_FUNDING_BLOCKS_LONG"
    CROWDED_SHORT       = "EXTREME_SHORT_FUNDING_BLOCKS_SHORT"
    WEAK_CONSENSUS      = "CONSENSUS_SCORE_BELOW_THRESHOLD"
    DIRECTION_CONFLICT  = "MAJORITY_CONFLICT_NO_CLEAR_DIRECTION"


# ─────────────────────────────────────────────
#  Trade Struct (JSON Output)
# ─────────────────────────────────────────────

@dataclass
class TradeStruct:
    """
    The unified output of the Head Agent.

    Passed downstream to SLTPEngine and ExecutionLayer.
    Fully JSON-serializable.
    """
    approved: bool
    direction: str              # 'LONG' | 'SHORT'
    consensus_score: float      # [-1, 1] — positive=bullish, negative=bearish
    composite_confidence: float # [0, 1]

    # Specialist breakdown
    chart_direction: str = "NEUTRAL"
    chart_confidence: float = 0.0
    chart_patterns: List[str] = field(default_factory=list)

    rl_direction: str = "HOLD"
    rl_leverage: float = 1.0
    rl_kelly: float = 0.25

    llm_direction: str = "NEUTRAL"
    llm_intent: str = "UNKNOWN"
    llm_narrative: str = ""

    # Veto info
    vetoed: bool = False
    veto_reason: Optional[str] = None

    # Metadata
    timestamp_ns: int = 0
    mid_price: float = 0.0

    # Predicted candle overlay (from ChartAgent)
    predicted_candles: List[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @property
    def is_actionable(self) -> bool:
        return self.approved and not self.vetoed and self.direction in ("LONG", "SHORT")


# ─────────────────────────────────────────────
#  Head Agent
# ─────────────────────────────────────────────

class HeadAgent:
    """
    Weighted Consensus Head Agent.

    Usage:
        head = HeadAgent(config)
        trade_struct = head.fuse(chart_signal, rl_decision, llm_signal, context)
        if trade_struct.is_actionable:
            execute(trade_struct)

    Args:
        config:               Hydra YAML config dict.
        weight_chart:         ChartAgent weight in consensus (default 0.35).
        weight_rl:            RLAgent weight (default 0.40).
        weight_llm:           LLMAgent weight (default 0.25).
        min_consensus_score:  Minimum |score| to approve a trade (default 0.25).
    """

    def __init__(
        self,
        config: dict,
        weight_chart: float = 0.35,
        weight_rl: float = 0.40,
        weight_llm: float = 0.25,
        min_consensus_score: float = 0.25,
    ) -> None:
        self._config = config
        self._w_chart = weight_chart
        self._w_rl = weight_rl
        self._w_llm = weight_llm
        self._min_score = min_consensus_score
        self._fused_count = 0
        self._vetoed_count = 0
        self._approved_count = 0

        logger.info(
            "HeadAgent | weights: chart=%.2f rl=%.2f llm=%.2f | min_score=%.2f",
            weight_chart, weight_rl, weight_llm, min_consensus_score,
        )

    def fuse(
        self,
        chart: Optional[ChartSignal],
        rl: Optional[RLDecision],
        llm: Optional[LLMSignal],
        mid_price: float = 0.0,
        heatmap_depth_ratio: float = 1.0,  # bid_depth / ask_depth
        funding_sentiment: str = "NEUTRAL",
        timestamp_ns: int = 0,
    ) -> TradeStruct:
        """
        Fuse three specialist signals into a single Trade Struct.

        Args:
            chart:               ChartSignal from ChartAgent.
            rl:                  RLDecision from RLAgent.
            llm:                 LLMSignal from LLMAgent.
            mid_price:           Current mid price.
            heatmap_depth_ratio: Bid depth / Ask depth ratio from heatmap.
            funding_sentiment:   'BULLISH'|'BEARISH'|'EXTREME_LONG'|'EXTREME_SHORT'|'NEUTRAL'
            timestamp_ns:        Bar timestamp.

        Returns:
            TradeStruct — fully populated, approved or vetoed.
        """
        self._fused_count += 1

        # ── Default safe values ────────────────
        chart = chart or ChartSignal(direction="NEUTRAL", confidence=0.33, patterns=[])
        rl = rl or RLDecision(direction="HOLD", leverage=1.0, kelly_fraction=0.25, confidence=0.33)
        llm = llm or LLMSignal(
            direction="NEUTRAL", confidence=0.5, narrative="No LLM signal.",
            intent="UNKNOWN", veto_recommendation=False
        )

        # ── Direction scoring ──────────────────
        def dir_score(direction: str) -> float:
            return {
                "LONG": 1.0, "SHORT": -1.0, "NEUTRAL": 0.0, "HOLD": 0.0
            }.get(direction, 0.0)

        chart_score = dir_score(chart.direction) * chart.confidence * self._w_chart
        rl_score    = dir_score(rl.direction)    * rl.confidence    * self._w_rl
        llm_score   = dir_score(llm.direction)   * llm.confidence   * self._w_llm

        consensus_score = chart_score + rl_score + llm_score
        composite_conf = (
            chart.confidence * self._w_chart
            + rl.confidence * self._w_rl
            + llm.confidence * self._w_llm
        )

        # ── Determine direction ────────────────
        if consensus_score > self._min_score:
            direction = "LONG"
        elif consensus_score < -self._min_score:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        # Build preliminary struct
        struct = TradeStruct(
            approved=False,
            direction=direction,
            consensus_score=consensus_score,
            composite_confidence=composite_conf,
            chart_direction=chart.direction,
            chart_confidence=chart.confidence,
            chart_patterns=chart.patterns,
            rl_direction=rl.direction,
            rl_leverage=rl.leverage,
            rl_kelly=rl.kelly_fraction,
            llm_direction=llm.direction,
            llm_intent=llm.intent,
            llm_narrative=llm.narrative,
            timestamp_ns=timestamp_ns,
            mid_price=mid_price,
            predicted_candles=chart.predicted_candles,
        )

        # ── Veto Logic ────────────────────────
        veto = self._apply_veto_rules(
            struct, rl, llm, funding_sentiment, heatmap_depth_ratio, direction
        )
        if veto:
            struct.vetoed = True
            struct.veto_reason = veto.value
            self._vetoed_count += 1
            logger.info(
                "🚫 Head Agent VETO | reason=%s | score=%.3f | dir=%s",
                veto.value, consensus_score, direction,
            )
            return struct

        # ── Approve ───────────────────────────
        if direction in ("LONG", "SHORT"):
            struct.approved = True
            self._approved_count += 1
            logger.info(
                "✅ Head Agent APPROVED | %s | score=%.3f | conf=%.2f | chart=%s rl=%s llm=%s",
                direction, consensus_score, composite_conf,
                chart.direction, rl.direction, llm.direction,
            )

        return struct

    def _apply_veto_rules(
        self,
        struct: TradeStruct,
        rl: RLDecision,
        llm: LLMSignal,
        funding_sentiment: str,
        heatmap_depth_ratio: float,
        direction: str,
    ) -> Optional[VetoReason]:
        """
        Apply veto rules in priority order. Returns first matching VetoReason.

        RULE-1: Heatmap shows heavy support + RL says SHORT
        RULE-2: LLM recommends veto
        RULE-3: All agents have low confidence
        RULE-4: Extreme long funding for new LONG entry
        RULE-5: Extreme short funding for new SHORT entry
        RULE-6: Consensus score below threshold
        RULE-7: No clear directional agreement
        """
        # RULE-1: RL SHORT but heavy support (whale accumulation zone)
        if rl.direction == "SHORT" and heatmap_depth_ratio > 1.5 and direction == "SHORT":
            return VetoReason.RL_SHORT_VS_SUPPORT

        # RULE-2: LLM explicitly vetoes
        if llm.veto_recommendation:
            return VetoReason.LLM_VETO

        # RULE-3: All agents below minimum confidence
        if (struct.chart_confidence < 0.50 and rl.confidence < 0.50 and llm.confidence < 0.50):
            return VetoReason.LOW_CONFIDENCE

        # RULE-4: Extreme long funding → no new longs (crowded trade)
        if direction == "LONG" and funding_sentiment == "EXTREME_LONG":
            return VetoReason.CROWDED_LONG

        # RULE-5: Extreme short funding → no new shorts (crowded trade)
        if direction == "SHORT" and funding_sentiment == "EXTREME_SHORT":
            return VetoReason.CROWDED_SHORT

        # RULE-6: Weak consensus score
        if direction != "NEUTRAL" and abs(struct.consensus_score) < self._min_score:
            return VetoReason.WEAK_CONSENSUS

        # RULE-7: No direction consensus
        if direction == "NEUTRAL":
            return VetoReason.DIRECTION_CONFLICT

        return None

    # ── Statistics ────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "fused": self._fused_count,
            "vetoed": self._vetoed_count,
            "approved": self._approved_count,
            "veto_rate": self._vetoed_count / max(self._fused_count, 1),
        }
