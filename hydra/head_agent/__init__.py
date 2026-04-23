"""
Hydra Head Agent — Genesis Consensus Engine
============================================
Fuses Chart, RL, and LLM specialist signals into a single Trade Struct.
Applies veto logic for conflicting signals.
"""
from hydra.head_agent.signal_fusion import HeadAgent, TradeStruct, VetoReason

__all__ = ["HeadAgent", "TradeStruct", "VetoReason"]
