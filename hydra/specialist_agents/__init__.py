"""
Hydra Specialist Agents — Genesis Swarm
========================================
Three specialist agents that feed the Head Agent:

  ChartAgent  — PatchTST-based pattern recognition (HH/HL, S/R, Whale Absorption)
  RLAgent     — FinRL PPO for dynamic leverage + Kelly sizing
  LLMAgent    — Claude 3.5 Sonnet for Whale Intent Inference
"""
from hydra.specialist_agents.chart_agent import ChartAgent, ChartSignal
from hydra.specialist_agents.rl_agent import RLAgent, RLDecision
from hydra.specialist_agents.llm_agent import LLMAgent, LLMSignal

__all__ = [
    "ChartAgent", "ChartSignal",
    "RLAgent", "RLDecision",
    "LLMAgent", "LLMSignal",
]
