"""
Genesis Engine — CLI Entry Point
==================================
Run modes:
  backtest  — 10,000-bar walk-forward validation
  paper     — Live paper trading (no real orders)
  validate  — Run WFO and generate PDF + Plotly report

Usage:
  python run_genesis.py backtest --bars 10000 --folds 5
  python run_genesis.py paper
  python run_genesis.py validate
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 output on Windows (cp1252 cannot encode emoji)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

Path("backtest/results").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest/results/genesis.log", mode="w", encoding="utf-8"),
    ],
)

logger = logging.getLogger("genesis")


def parse_args():
    p = argparse.ArgumentParser(description="Hydra Genesis Engine")
    p.add_argument("mode", choices=["backtest", "paper", "live", "validate"],
                   help="Execution mode")
    p.add_argument("--config", default="config/genesis_config.yaml",
                   help="Path to genesis_config.yaml")
    p.add_argument("--bars", type=int, default=10000, help="Number of bars (backtest)")
    p.add_argument("--folds", type=int, default=5, help="WFO folds (validate)")
    return p.parse_args()


async def run_backtest(config_path: str, n_bars: int) -> None:
    from hydra.genesis_engine import GenesisEngine
    engine = GenesisEngine.from_config(config_path, mode="backtest")
    stats = await engine.run(n_bars=n_bars)

    # Generate chart
    try:
        from validation.walk_forward import generate_plotly_chart
        prices = [t.entry_price for t in engine.trade_history] or [65000.0]
        generate_plotly_chart(engine.trade_history, prices)
    except Exception as e:
        logger.warning("Chart generation failed: %s", e)

    logger.info("Backtest complete. PF target >1.8 | Sharpe target >1.0")


async def run_validate(config_path: str, bars: int, folds: int) -> None:
    from validation.walk_forward import run_walk_forward, generate_pdf_report
    Path("backtest/results").mkdir(parents=True, exist_ok=True)
    report = await run_walk_forward(config_path, bars, folds)
    generate_pdf_report(report)


async def run_paper(config_path: str) -> None:
    from hydra.genesis_engine import GenesisEngine
    engine = GenesisEngine.from_config(config_path, mode="paper")
    await engine.run()


async def main():
    Path("backtest/results").mkdir(parents=True, exist_ok=True)
    args = parse_args()

    logger.info("🌀 Hydra Genesis Engine | mode=%s", args.mode)

    if args.mode == "backtest":
        await run_backtest(args.config, args.bars)
    elif args.mode == "validate":
        await run_validate(args.config, args.bars, args.folds)
    elif args.mode in ("paper", "live"):
        await run_paper(args.config)


if __name__ == "__main__":
    asyncio.run(main())
