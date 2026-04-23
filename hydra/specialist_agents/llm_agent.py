"""
LLM Agent — Claude 3.5 Sonnet Whale Intent Inference
======================================================
Specialist Agent #3 in the Hydra Swarm.

Uses the Anthropic Claude API to reason about L2 order book imbalances
and link microstructure signals to high-level market narratives.

Primary Function: "Whale Intent Inference"
  Given:
    - L2 imbalances (large bid/ask walls)
    - Heatmap support/resistance clusters
    - Funding rate & OI context
    - Recent chart patterns
  The LLM infers:
    - Is this a genuine accumulation or a spoofed wall?
    - What is the institutional intent behind this order flow?
    - Does the narrative support or contradict the RL direction?

Rate Limiting:
  - LLM calls are expensive; defaults to max 1 call per 60s.
  - Falls back to a lightweight rule-based reasoner when rate-limited.
  - In backtest mode, the LLM is bypassed entirely (uses rules).

API Key:
  Set env var ANTHROPIC_API_KEY or provide in config['llm']['api_key'].
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  LLM Signal
# ─────────────────────────────────────────────

@dataclass
class LLMSignal:
    """Output from the LLM Whale Intent Inference agent."""
    direction: str            # 'LONG' | 'SHORT' | 'NEUTRAL'
    confidence: float         # [0, 1]
    narrative: str            # Plain-English explanation
    intent: str               # 'ACCUMULATION' | 'DISTRIBUTION' | 'SPOOF' | 'GENUINE' | 'UNKNOWN'
    veto_recommendation: bool  # True if LLM recommends vetoing the trade
    timestamp_ns: int = 0
    source: str = "llm_agent"
    latency_ms: float = 0.0

    @property
    def is_actionable(self) -> bool:
        return self.direction != "NEUTRAL" and self.confidence >= 0.55


# ─────────────────────────────────────────────
#  Prompt Builder
# ─────────────────────────────────────────────

class WhaleIntentPromptBuilder:
    """Constructs structured prompts for whale intent inference."""

    SYSTEM_PROMPT = """You are a professional crypto market microstructure analyst specializing in 
institutional order flow and whale activity detection. Your role is to analyze L2 order book data, 
funding rates, and chart patterns to infer the intent behind large orders.

Respond ONLY with a valid JSON object. No markdown, no extra text.
JSON schema:
{
  "direction": "LONG" | "SHORT" | "NEUTRAL",
  "confidence": 0.0–1.0,
  "intent": "ACCUMULATION" | "DISTRIBUTION" | "SPOOF" | "GENUINE" | "UNKNOWN",
  "narrative": "2-3 sentence explanation",
  "veto_recommendation": true | false
}"""

    @staticmethod
    def build(
        mid_price: float,
        bid_wall: float,
        ask_wall: float,
        cum_bid_depth_usd: float,
        cum_ask_depth_usd: float,
        support_levels: List[float],
        resistance_levels: List[float],
        funding_rate: float,
        open_interest: float,
        oi_delta: float,
        patterns: List[str],
        rl_direction: str,
    ) -> str:
        depth_ratio = cum_bid_depth_usd / max(cum_ask_depth_usd, 1)
        support_str = ", ".join(f"${s:,.0f}" for s in support_levels[:3]) or "none"
        resistance_str = ", ".join(f"${r:,.0f}" for r in resistance_levels[:3]) or "none"
        pattern_str = ", ".join(patterns) if patterns else "none detected"

        return f"""Analyze this L2 order book snapshot for whale intent:

PRICE: ${mid_price:,.2f}
BID WALL: ${bid_wall:,.2f} (largest single bid level)
ASK WALL: ${ask_wall:,.2f} (largest single ask level)
BID DEPTH: ${cum_bid_depth_usd:,.0f} USD
ASK DEPTH: ${cum_ask_depth_usd:,.0f} USD
DEPTH RATIO (bid/ask): {depth_ratio:.2f}

HEATMAP SUPPORT: {support_str}
HEATMAP RESISTANCE: {resistance_str}

FUNDING RATE: {funding_rate*100:.4f}% (annualized: {funding_rate*3*365*100:.1f}%)
OPEN INTEREST: ${open_interest:,.0f}
OI CHANGE: {'+' if oi_delta >= 0 else ''}{oi_delta:,.0f} USD

CHART PATTERNS: {pattern_str}
RL AGENT SIGNAL: {rl_direction}

Infer the whale intent and provide your trading recommendation."""


# ─────────────────────────────────────────────
#  LLM Agent
# ─────────────────────────────────────────────

class LLMAgent:
    """
    Claude 3.5 Sonnet Whale Intent Inference Specialist.

    Args:
        config:          Hydra YAML config dict.
        mode:            'backtest' | 'paper' | 'live'
        min_interval_s:  Minimum seconds between LLM API calls (rate limit guard).
    """

    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        config: dict,
        mode: str = "backtest",
        min_interval_s: float = 60.0,
    ) -> None:
        self._config = config
        self._mode = mode
        self._min_interval_s = min_interval_s
        self._last_call_ts: float = 0.0
        self._last_signal: Optional[LLMSignal] = None
        self._call_count = 0

        llm_cfg = config.get("llm", {})
        self._api_key = llm_cfg.get("api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = llm_cfg.get("model", self.DEFAULT_MODEL)
        self._max_tokens = llm_cfg.get("max_tokens", 256)
        self._client = None

        if self._api_key and mode != "backtest":
            self._init_client()

        logger.info(
            "LLMAgent initialized | model=%s | mode=%s | api_key=%s",
            self._model, mode, "SET" if self._api_key else "NOT SET",
        )

    def _init_client(self) -> None:
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic(api_key=self._api_key)
            logger.info("LLMAgent: Anthropic client initialized.")
        except ImportError:
            logger.warning("LLMAgent: anthropic SDK not installed. pip install anthropic")
        except Exception as e:
            logger.warning("LLMAgent: could not initialize client: %s", e)

    async def process(
        self,
        mid_price: float,
        bid_wall: float,
        ask_wall: float,
        cum_bid_depth: float,
        cum_ask_depth: float,
        support_levels: List[float],
        resistance_levels: List[float],
        funding_rate: float,
        open_interest: float,
        oi_delta: float,
        patterns: List[str],
        rl_direction: str,
        timestamp_ns: int = 0,
    ) -> LLMSignal:
        """
        Run whale intent inference.

        If rate-limited, in backtest mode, or API unavailable — returns a
        rule-based signal. Otherwise calls Claude 3.5 Sonnet.
        """
        now = time.time()
        rate_limited = (now - self._last_call_ts) < self._min_interval_s
        can_call_api = (
            self._client is not None
            and self._mode != "backtest"
            and self._api_key
            and not rate_limited
        )

        if can_call_api:
            signal = await self._llm_inference(
                mid_price, bid_wall, ask_wall, cum_bid_depth, cum_ask_depth,
                support_levels, resistance_levels, funding_rate, open_interest,
                oi_delta, patterns, rl_direction, timestamp_ns,
            )
        else:
            signal = self._rule_based_inference(
                mid_price, bid_wall, ask_wall, cum_bid_depth, cum_ask_depth,
                support_levels, resistance_levels, funding_rate, oi_delta,
                patterns, rl_direction, timestamp_ns,
            )

        self._last_signal = signal
        return signal

    async def _llm_inference(
        self,
        mid_price, bid_wall, ask_wall, cum_bid_depth, cum_ask_depth,
        support_levels, resistance_levels, funding_rate, open_interest,
        oi_delta, patterns, rl_direction, timestamp_ns,
    ) -> LLMSignal:
        prompt = WhaleIntentPromptBuilder.build(
            mid_price=mid_price, bid_wall=bid_wall, ask_wall=ask_wall,
            cum_bid_depth_usd=cum_bid_depth, cum_ask_depth_usd=cum_ask_depth,
            support_levels=support_levels, resistance_levels=resistance_levels,
            funding_rate=funding_rate, open_interest=open_interest,
            oi_delta=oi_delta, patterns=patterns, rl_direction=rl_direction,
        )

        t0 = time.time()
        try:
            # Run blocking Anthropic call in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=WhaleIntentPromptBuilder.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
            latency_ms = (time.time() - t0) * 1000
            self._last_call_ts = time.time()
            self._call_count += 1

            raw_text = response.content[0].text.strip()
            parsed = json.loads(raw_text)

            signal = LLMSignal(
                direction=parsed.get("direction", "NEUTRAL"),
                confidence=float(parsed.get("confidence", 0.5)),
                narrative=parsed.get("narrative", ""),
                intent=parsed.get("intent", "UNKNOWN"),
                veto_recommendation=bool(parsed.get("veto_recommendation", False)),
                timestamp_ns=timestamp_ns,
                latency_ms=latency_ms,
            )
            logger.info(
                "LLMAgent: %s | conf=%.2f | intent=%s | latency=%.0fms",
                signal.direction, signal.confidence, signal.intent, latency_ms,
            )
            return signal

        except json.JSONDecodeError as e:
            logger.warning("LLMAgent: JSON parse error: %s", e)
        except Exception as e:
            logger.warning("LLMAgent: API error: %s", e)

        # Fallback on any error
        return self._rule_based_inference(
            mid_price, bid_wall, ask_wall, cum_bid_depth, cum_ask_depth,
            support_levels, resistance_levels, funding_rate, oi_delta,
            patterns, rl_direction, timestamp_ns,
        )

    @staticmethod
    def _rule_based_inference(
        mid_price, bid_wall, ask_wall, cum_bid_depth, cum_ask_depth,
        support_levels, resistance_levels, funding_rate, oi_delta,
        patterns, rl_direction, timestamp_ns,
    ) -> LLMSignal:
        """Heuristic whale intent without LLM API."""
        depth_ratio = cum_bid_depth / max(cum_ask_depth, 1)
        near_support = any(abs(s - mid_price) / mid_price < 0.003 for s in support_levels)
        near_resistance = any(abs(r - mid_price) / mid_price < 0.003 for r in resistance_levels)

        # Whale absorption: heavy bid wall near support, rising OI
        if depth_ratio > 1.5 and near_support and oi_delta > 0 and "WHALE_ABSORPTION" in patterns:
            return LLMSignal(
                direction="LONG", confidence=0.75, intent="ACCUMULATION",
                narrative=(
                    f"Heavy bid depth (ratio {depth_ratio:.1f}x) at support ${support_levels[0]:,.0f} "
                    f"with rising OI (+{oi_delta:,.0f}) suggests institutional accumulation. "
                    f"Bid wall at ${bid_wall:,.0f} indicates genuine demand."
                ),
                veto_recommendation=False, timestamp_ns=timestamp_ns,
            )

        # Distribution: heavy ask wall near resistance, extreme funding
        if depth_ratio < 0.7 and near_resistance and funding_rate > 0.001:
            return LLMSignal(
                direction="SHORT", confidence=0.70, intent="DISTRIBUTION",
                narrative=(
                    f"Ask wall at ${ask_wall:,.0f} near resistance with extreme funding rate "
                    f"({funding_rate*100:.3f}%) suggests crowded longs being distributed. "
                    f"Veto any long signal."
                ),
                veto_recommendation=(rl_direction == "LONG"),
                timestamp_ns=timestamp_ns,
            )

        # Potential spoof: large wall but opposite to RL signal
        if depth_ratio > 2.0 and rl_direction == "SHORT":
            return LLMSignal(
                direction="NEUTRAL", confidence=0.55, intent="SPOOF",
                narrative=(
                    f"Very high bid/ask ratio ({depth_ratio:.1f}x) but RL signals SHORT. "
                    f"Large bid wall may be a spoof — recommend caution."
                ),
                veto_recommendation=True, timestamp_ns=timestamp_ns,
            )

        # Default: neutral, agree with RL
        return LLMSignal(
            direction=rl_direction, confidence=0.50, intent="UNKNOWN",
            narrative="No strong whale signal detected. Deferring to RL decision.",
            veto_recommendation=False, timestamp_ns=timestamp_ns,
        )

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_signal(self) -> Optional[LLMSignal]:
        return self._last_signal
