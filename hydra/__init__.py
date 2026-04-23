"""
Hydra Genesis — Multi-Agent Swarm Trading Engine
=================================================
Top-level package. Exports the GenesisEngine for direct use.

Usage:
    from hydra import GenesisEngine
    import asyncio

    engine = GenesisEngine.from_config('config/genesis_config.yaml', mode='backtest')
    stats = asyncio.run(engine.run(n_bars=10000))
"""
from hydra.genesis_engine import GenesisEngine, GenesisStats, GenesisTrade, GenesisState

__version__ = "2.0.0-genesis"
__all__ = ["GenesisEngine", "GenesisStats", "GenesisTrade", "GenesisState"]
