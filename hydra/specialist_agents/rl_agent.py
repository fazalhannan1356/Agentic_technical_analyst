"""
RL Agent — FinRL PPO for Dynamic Leverage + Kelly Criterion Sizing
===================================================================
Specialist Agent #2 in the Hydra Swarm.

Uses a Proximal Policy Optimization (PPO) network to:
  1. Decide trade direction based on market state features.
  2. Output dynamic leverage (1×–20×) via continuous action.
  3. Position sizing via fractional Kelly Criterion.

State Space (observation):
  - Fractionally-differenced mid price (preserves memory)
  - OFI, book pressure, spread_bps, VWAP deviation
  - Funding rate, OI delta, estimated liquidations
  - Running PnL, current drawdown, win rate (account state)

Action Space:
  - Discrete: 0=HOLD, 1=LONG, 2=SHORT
  - Continuous: leverage ∈ [1.0, 20.0]

Reward Function:
  r_t = Sharpe-adjusted PnL - drawdown_penalty - fee_cost
  
This module implements the RL environment + a lightweight PPO actor-critic
that can be trained without FinRL as an external dep (FinRL is optional).
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────
#  RL Decision
# ─────────────────────────────────────────────

@dataclass
class RLDecision:
    direction: str     # 'LONG' | 'SHORT' | 'HOLD'
    leverage: float    # 1.0–20.0
    kelly_fraction: float   # 0.0–0.5
    confidence: float  # [0, 1] — probability of the chosen action
    timestamp_ns: int = 0
    source: str = "rl_agent"

    @property
    def position_fraction(self) -> float:
        """Recommended fraction of balance to allocate."""
        return min(self.kelly_fraction, 0.5)


# ─────────────────────────────────────────────
#  PPO Actor-Critic Network
# ─────────────────────────────────────────────

if TORCH_AVAILABLE:
    class PPOActorCritic(nn.Module):
        """
        Shared-backbone Actor-Critic for PPO.

        Actor head:
          - Discrete: softmax over [HOLD, LONG, SHORT]
          - Continuous: sigmoid → leverage ∈ [1, 20]

        Critic head:
          - Value estimate V(s)
        """
        def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.action_head = nn.Linear(hidden_dim, 3)   # HOLD / LONG / SHORT logits
            self.leverage_head = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.value_head = nn.Linear(hidden_dim, 1)

        def forward(self, state: "torch.Tensor"):
            feat = self.backbone(state)
            action_logits = self.action_head(feat)
            leverage_raw = self.leverage_head(feat)          # [0, 1]
            value = self.value_head(feat)
            return action_logits, leverage_raw, value

        def act(self, state: "torch.Tensor"):
            """Sample an action. Returns (action_idx, leverage, log_prob, value)."""
            logits, lev_raw, value = self.forward(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            leverage = 1.0 + lev_raw.squeeze() * 19.0  # [1, 20]
            return action.item(), float(leverage.item()), float(log_prob.item()), float(value.item())


# ─────────────────────────────────────────────
#  RL Agent
# ─────────────────────────────────────────────

class RLAgent:
    """
    PPO-based Reinforcement Learning specialist agent.

    Produces a leverage-tagged directional decision from market state features.
    Falls back to rule-based Kelly sizing if PyTorch is unavailable.

    Args:
        config:     Hydra YAML config dict.
        model_path: Optional pretrained PPO model (.pt).
    """

    STATE_DIM = 15   # See _build_state_vector

    def __init__(self, config: dict, model_path: Optional[str] = None) -> None:
        self._config = config
        self._kelly_base = config.get("risk", {}).get("kelly_fraction", 0.25)
        self._max_leverage = config.get("leverage", {}).get("tier_3_leverage", 10)
        self._model: Optional["PPOActorCritic"] = None
        self._trade_pnl_history: deque = deque(maxlen=100)

        if TORCH_AVAILABLE:
            self._model = PPOActorCritic(state_dim=self.STATE_DIM)
            if model_path:
                try:
                    state = torch.load(model_path, map_location="cpu")
                    self._model.load_state_dict(state)
                    self._model.eval()
                    logger.info("RLAgent: loaded PPO model from %s", model_path)
                except Exception as e:
                    logger.warning("RLAgent: could not load model (%s)", e)
        else:
            logger.warning("RLAgent: PyTorch unavailable — heuristic mode.")

        logger.info("RLAgent initialized | STATE_DIM=%d | torch=%s", self.STATE_DIM, TORCH_AVAILABLE)

    def process(
        self,
        ofi: float,
        book_pressure: float,
        spread_bps: float,
        price_to_vwap: float,
        funding_rate: float,
        oi_delta: float,
        est_liq: float,
        frac_diff_price: float,
        running_pnl: float,
        drawdown: float,
        win_rate: float,
        balance: float,
        timestamp_ns: int = 0,
    ) -> RLDecision:
        """
        Produce an RL decision from the current market + account state.

        Returns:
            RLDecision with direction, leverage, kelly_fraction, confidence.
        """
        state = self._build_state_vector(
            ofi, book_pressure, spread_bps, price_to_vwap,
            funding_rate, oi_delta, est_liq, frac_diff_price,
            running_pnl, drawdown, win_rate, balance,
        )

        if TORCH_AVAILABLE and self._model is not None:
            return self._nn_decision(state, timestamp_ns)
        else:
            return self._heuristic_decision(state, timestamp_ns)

    def record_trade_pnl(self, pnl: float) -> None:
        """Feed trade outcome back to the agent (for Kelly update)."""
        self._trade_pnl_history.append(pnl)

    # ── State Building ────────────────────────

    @staticmethod
    def _build_state_vector(
        ofi, book_pressure, spread_bps, price_to_vwap,
        funding_rate, oi_delta, est_liq, frac_diff_price,
        running_pnl, drawdown, win_rate, balance,
    ) -> np.ndarray:
        """Normalise and assemble the RL state vector (dim=15)."""
        # Clip extreme values to ±5 sigma
        def clip(x, lo=-5.0, hi=5.0):
            return float(np.clip(x, lo, hi))

        normalized_balance = clip(math.log1p(max(balance, 1)) / 10.0)
        normalized_oi_delta = clip(oi_delta / 1e8)
        normalized_liq = clip(est_liq / 1e8)

        return np.array([
            clip(ofi / 100),               # 0: Order Flow Imbalance
            clip(book_pressure - 0.5),     # 1: Book pressure centred
            clip(spread_bps / 20),         # 2: Spread in BPS
            clip(price_to_vwap * 100),     # 3: Price deviation from VWAP
            clip(funding_rate * 1000),     # 4: Funding rate scaled
            normalized_oi_delta,           # 5: OI change
            normalized_liq,                # 6: Est. liquidations
            clip(frac_diff_price),         # 7: Fractionally differenced price
            clip(running_pnl / 1000),      # 8: Running PnL
            clip(drawdown * 10),           # 9: Current drawdown
            clip(win_rate - 0.5),          # 10: Win rate centred
            normalized_balance,            # 11: Account balance
            0.0, 0.0, 0.0,                 # 12-14: Reserved
        ], dtype=np.float32)

    def _nn_decision(self, state: np.ndarray, ts_ns: int) -> RLDecision:
        """PPO actor forward pass."""
        try:
            state_t = torch.from_numpy(state).unsqueeze(0)  # [1, STATE_DIM]
            with torch.no_grad():
                action_idx, leverage, log_prob, value = self._model.act(state_t)

            direction_map = {0: "HOLD", 1: "LONG", 2: "SHORT"}
            direction = direction_map[action_idx]
            confidence = float(np.exp(log_prob))  # higher log_prob → higher confidence

            leverage = float(np.clip(leverage, 1.0, self._max_leverage))
            kelly = self._compute_kelly()

            return RLDecision(
                direction=direction,
                leverage=leverage,
                kelly_fraction=kelly,
                confidence=min(confidence, 0.99),
                timestamp_ns=ts_ns,
            )
        except Exception as e:
            logger.warning("RLAgent NN decision error: %s", e)
            return self._heuristic_decision(state, ts_ns)

    def _heuristic_decision(self, state: np.ndarray, ts_ns: int) -> RLDecision:
        """
        Momentum-first heuristic.

        Feature map (from _build_state_vector):
          state[0] = OFI / 100           → order flow imbalance
          state[1] = book_pressure - 0.5 → positive = more bids
          state[4] = funding * 1000      → positive = longs pay shorts
          state[7] = frac_diff_price     → PRIMARY: stationary trend signal

        In regime-switching markets, the fractionally-differenced price
        accumulates directional drift within each regime and is the
        strongest single predictor of near-term direction.
        """
        frac_diff = float(state[7])     # Primary: stationary trend signal
        book_press = float(state[1])    # Confirmation: bid/ask imbalance
        ofi = float(state[0])           # Confirmation: order flow
        funding = float(state[4])       # Contrarian filter

        # Composite score (frac_diff dominates at 60%, book/OFI split 40%)
        score = frac_diff * 0.60 + book_press * 0.25 + ofi * 0.15

        # Contrarian funding nudge (overcrowded positions often reverse)
        score -= funding * 0.05

        THRESHOLD = 0.18  # Calibrated for regime-switching vol profile

        if score > THRESHOLD:
            conf = float(np.clip(0.52 + abs(score) * 0.5, 0.52, 0.92))
            direction = "LONG"
        elif score < -THRESHOLD:
            conf = float(np.clip(0.52 + abs(score) * 0.5, 0.52, 0.92))
            direction = "SHORT"
        else:
            direction, conf = "HOLD", 0.38

        leverage = float(np.clip(3.0 + abs(score) * 8.0, 2.0, self._max_leverage))
        kelly = self._compute_kelly()

        return RLDecision(
            direction=direction,
            leverage=leverage,
            kelly_fraction=kelly,
            confidence=conf,
            timestamp_ns=ts_ns,
        )


    def _compute_kelly(self) -> float:
        """Fractional Kelly based on recent trade history."""
        history = list(self._trade_pnl_history)
        if len(history) < 5:
            return self._kelly_base

        wins = [p for p in history if p > 0]
        losses = [p for p in history if p < 0]
        if not wins or not losses:
            return self._kelly_base

        p = len(wins) / len(history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        if avg_loss == 0:
            return self._kelly_base

        b = avg_win / avg_loss
        kelly = (b * p - (1 - p)) / b
        # Fractional Kelly (1/4 Kelly)
        return float(np.clip(kelly * 0.25, 0.02, 0.5))

    def train_ppo(
        self,
        env_data: List[Tuple],
        n_epochs: int = 10,
        lr: float = 3e-4,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Simplified PPO update on a collected rollout buffer.

        env_data: List of (state, action, reward, log_prob, value, done) tuples.
        """
        if not TORCH_AVAILABLE or self._model is None:
            logger.warning("RLAgent.train_ppo: PyTorch not available.")
            return

        import torch.optim as optim

        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        clip_eps = 0.2
        gamma, lam = 0.99, 0.95

        # Unpack rollout
        states, actions, rewards, old_log_probs, values, dones = zip(*env_data)
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        old_lp_t = torch.tensor(old_log_probs, dtype=torch.float32)

        # Compute returns & advantages (GAE)
        returns, adv = [], []
        R, A = 0.0, 0.0
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            R = r + gamma * R * (1 - d)
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)
        adv_t = returns_t - values_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for epoch in range(n_epochs):
            logits, _, new_values = self._model(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(new_values.squeeze(), returns_t)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._model.parameters(), 0.5)
            optimizer.step()

        logger.info("PPO training complete | final_loss=%.4f", loss.item())
        if save_path:
            torch.save(self._model.state_dict(), save_path)
            logger.info("RLAgent PPO model saved to %s", save_path)
