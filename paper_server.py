"""
Hydra Genesis — Paper Trading Server
======================================
Runs GenesisEngine with real Binance data and streams all events to a
live browser dashboard via WebSocket.

Usage:
    python paper_server.py

Ports:
    http://localhost:8766  — Live dashboard
    ws://localhost:8765    — WebSocket feed
"""
from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import logging
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Set

import yaml

sys.path.insert(0, str(Path(__file__).parent))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

Path("backtest/results").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest/results/paper.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("paper_server")

# ── Import Genesis ────────────────────────────────────────────────
from hydra.genesis_engine import GenesisEngine
from hydra.head_agent.signal_fusion import TradeStruct

# ── Global client registry ────────────────────────────────────────
CLIENTS: Set = set()


async def broadcast(data: dict) -> None:
    if not CLIENTS:
        return
    msg = json.dumps(data, default=str)
    dead = set()
    for ws in list(CLIENTS):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    CLIENTS -= dead


# ── Instrumented Genesis Engine ───────────────────────────────────

class PaperGenesisEngine(GenesisEngine):
    """GenesisEngine with WebSocket event hooks for the dashboard."""

    async def _process_bar(self, snapshot) -> None:
        # Broadcast price tick before processing
        await broadcast({
            "type": "price",
            "price": round(snapshot.mid_price, 2),
            "bid":   round(snapshot.best_bid, 2),
            "ask":   round(snapshot.best_ask, 2),
            "ts":    time.time(),
        })
        await super()._process_bar(snapshot)
        # Broadcast updated stats after each bar
        s = self.stats
        wins   = [t.pnl for t in self._trade_history if t.pnl and t.pnl > 0]
        losses = [abs(t.pnl) for t in self._trade_history if t.pnl and t.pnl < 0]
        pf = sum(wins) / max(sum(losses), 1e-8) if losses else 0.0
        await broadcast({
            "type":    "stats",
            "balance": round(self._balance, 2),
            "trades":  s.total_trades,
            "win_rate": round(s.win_rate * 100, 1),
            "profit_factor": round(pf, 3),
            "drawdown": round(s.max_drawdown * 100, 2),
            "signals":  s.signals_generated,
            "vetoed":   s.signals_vetoed,
        })

    async def _evaluate_and_execute(self, ts: TradeStruct, snapshot) -> None:
        prev = self._current_trade
        await super()._evaluate_and_execute(ts, snapshot)
        # Signal event
        await broadcast({
            "type":       "signal",
            "direction":  ts.direction,
            "approved":   ts.approved,
            "vetoed":     ts.vetoed,
            "veto_reason": ts.veto_reason,
            "score":      round(ts.consensus_score, 3),
            "confidence": round(ts.composite_confidence, 3),
            "chart_dir":  ts.chart_direction,
            "chart_conf": round(ts.chart_confidence, 3),
            "rl_dir":     ts.rl_direction,
            "llm_dir":    ts.llm_direction,
            "ts":         time.time(),
        })
        # New trade opened
        if self._current_trade and self._current_trade is not prev:
            t = self._current_trade
            await broadcast({
                "type":        "trade_open",
                "trade_id":    t.trade_id,
                "direction":   t.direction,
                "entry":       round(t.entry_price, 2),
                "sl":          round(t.stop_loss, 2),
                "tp":          round(t.take_profit, 2),
                "position_usd": round(t.position_usd, 2),
                "leverage":    t.leverage,
                "ts":          time.time(),
            })

    async def _close_trade(self, exit_price: float, reason: str, ts_ns: int) -> None:
        trade = self._current_trade
        await super()._close_trade(exit_price, reason, ts_ns)
        if trade:
            await broadcast({
                "type":      "trade_close",
                "trade_id":  trade.trade_id,
                "direction": trade.direction,
                "exit":      round(exit_price, 2),
                "pnl":       round(trade.pnl or 0, 4),
                "reason":    reason,
                "ts":        time.time(),
            })


# ── WebSocket handler ────────────────────────────────────────────

async def ws_handler(websocket, path=None) -> None:
    CLIENTS.add(websocket)
    logger.info("Dashboard connected | clients=%d", len(CLIENTS))
    try:
        await websocket.send(json.dumps({
            "type": "hello",
            "msg":  "Hydra Genesis Paper Trading — Connected",
        }))
        await websocket.wait_closed()
    except Exception:
        pass
    finally:
        CLIENTS.discard(websocket)
        logger.info("Dashboard disconnected | clients=%d", len(CLIENTS))


# ── HTTP server for dashboard HTML ──────────────────────────────

def start_http_server(port: int = 8766) -> None:
    dashboard_dir = Path(__file__).parent / "dashboard"
    dashboard_dir.mkdir(exist_ok=True)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(dashboard_dir), **kw)
        def log_message(self, *a):
            pass

    srv = HTTPServer(("0.0.0.0", port), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    logger.info("Dashboard: http://localhost:%d", port)


# ── Feed builder ─────────────────────────────────────────────────

def build_paper_feed(config: dict):
    try:
        from hydra.data_engineer.binance_feed import BinancePaperFeed
        return BinancePaperFeed(config)
    except Exception as e:
        logger.warning("Binance feed unavailable (%s) — using synthetic feed", e)
        from hydra.data_engineer.l2_feed_genesis import L2FeedGenesis
        return L2FeedGenesis(config)


# ── Main ─────────────────────────────────────────────────────────

async def main() -> None:
    config_path = "config/genesis_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # HTTP dashboard server (background thread)
    start_http_server(8766)

    # WebSocket server
    try:
        import websockets
        ws_server = await websockets.serve(ws_handler, "0.0.0.0", 8765)
        logger.info("WebSocket: ws://localhost:8765")
    except ImportError:
        logger.error("websockets not installed. Run: pip install websockets")
        return

    # Paper engine with Binance real data
    engine = PaperGenesisEngine(config, mode="paper")

    # Monkey-patch feed so engine uses Binance
    _orig_build = engine._build_feed
    engine._build_feed = lambda: build_paper_feed(config)

    logger.info("=" * 55)
    logger.info("  HYDRA GENESIS — PAPER TRADING LIVE")
    logger.info("  Dashboard  : http://localhost:8766")
    logger.info("  WebSocket  : ws://localhost:8765")
    logger.info("  Press Ctrl+C to stop")
    logger.info("=" * 55)

    try:
        await engine.run()
    except KeyboardInterrupt:
        pass
    finally:
        ws_server.close()
        logger.info("Paper trading stopped.")


if __name__ == "__main__":
    asyncio.run(main())
