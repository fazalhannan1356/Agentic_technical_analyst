# Hydra Genesis Engine — Multi-Agent Swarm Trading System

> **Genesis Architecture** — L1/L2 Fusion + Agentic Specialist Swarm

## Architecture Overview

```
MarketDataBus (Redis/InProcess)
         │
    ┌────▼───────────────────────────────────────────┐
    │        HYPER-INGESTION LAYER                    │
    │  L2FeedGenesis (100-level LOB)                 │
    │  LiquidityHeatmap (VAP Whale Clusters)          │
    │  ContextFeed (Funding / OI / Liquidations)      │
    │  FractionalDifferentiator (d=0.4 FFD)          │
    └────┬───────────────────────────────────────────┘
         │
    ┌────▼───────────────────────────────────────────┐
    │        SPECIALIST AGENT SWARM                   │
    │  ChartAgent  (PatchTST) — HH/HL, S/R, Whale   │
    │  RLAgent     (PPO)      — Dynamic Leverage      │
    │  LLMAgent    (Claude)   — Whale Intent          │
    └────┬───────────────────────────────────────────┘
         │
    ┌────▼───────────────────────────────────────────┐
    │        HEAD AGENT (Consensus Fusion)            │
    │  Weighted: Chart=35% | RL=40% | LLM=25%        │
    │  7-Rule Veto Logic (Heatmap / Funding / RL)    │
    └────┬───────────────────────────────────────────┘
         │
    ┌────▼───────────────────────────────────────────┐
    │        OUTPUTS                                  │
    │  TradeStruct (JSON) → SLTPEngine → Execution   │
    │  Plotly Charts + PDF Reports                    │
    │  Walk-Forward Validation (10,000 bars)          │
    └────────────────────────────────────────────────┘
```

## Module Structure

```
hydra/
├── __init__.py                    # Exports GenesisEngine
├── genesis_engine.py              # Main orchestrator
├── data_engineer/
│   ├── market_bus.py              # Redis Pub/Sub + InProcess fallback
│   ├── frac_diff.py               # Fractional Differentiation (d=0.4 FFD)
│   ├── l2_feed_genesis.py         # 100-level L2 feed
│   ├── heatmap.py                 # Rolling Volume-at-Price heatmap
│   └── context_feed.py            # Funding Rate / OI / Liquidations
├── specialist_agents/
│   ├── chart_agent.py             # PatchTST pattern recognition
│   ├── rl_agent.py                # PPO dynamic leverage + Kelly sizing
│   └── llm_agent.py               # Claude 3.5 Sonnet Whale Intent Inference
├── head_agent/
│   └── signal_fusion.py           # Weighted consensus + 7-rule veto
└── risk_manager/                  # Existing (fee_guard, sl_tp, leverage)
config/
└── genesis_config.yaml            # Full configuration
validation/
└── walk_forward.py                # WFO + Plotly + PDF reporting
run_genesis.py                     # CLI entry point
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run 10,000-bar backtest
python run_genesis.py backtest --bars 10000

# Run walk-forward validation + PDF report
python run_genesis.py validate --bars 10000 --folds 5

# Paper trading (live data, no real orders)
python run_genesis.py paper
```

## Configuration

Edit `config/genesis_config.yaml`:

```yaml
# LLM Agent (set your Claude API key)
llm:
  api_key: "your_anthropic_api_key"

# Redis Bus (optional)
bus:
  redis_enabled: true
  redis_host: localhost
```

## Specialist Agent Details

### ChartAgent (PatchTST)
- **Input**: Rolling price + feature window (64 bars, 8 features/bar)
- **Architecture**: PatchEmbedding → TransformerEncoder → Classification + Forecast head
- **Patterns**: HH/HL, LH/LL, Whale Absorption, Consolidation, S/R zones
- **Output**: `ChartSignal(direction, confidence, patterns, predicted_candles)`

### RLAgent (PPO)
- **State**: 15-dim vector (OFI, book pressure, funding rate, frac-diff price, account state)
- **Action**: Discrete (HOLD/LONG/SHORT) + Continuous leverage [1×–20×]
- **Reward**: Sharpe-adjusted PnL - drawdown penalty - fee cost
- **Sizing**: Fractional Kelly Criterion (1/4 Kelly)
- **Output**: `RLDecision(direction, leverage, kelly_fraction, confidence)`

### LLMAgent (Claude 3.5 Sonnet)
- **Task**: Whale Intent Inference from L2 imbalances + context
- **Output**: `LLMSignal(direction, confidence, intent, narrative, veto_recommendation)`
- **Rate Limit**: 1 call per 60s (configurable)
- **Fallback**: Rule-based heuristics when API unavailable/rate-limited

### HeadAgent (Consensus Fusion)
- **Weights**: Chart=35%, RL=40%, LLM=25%
- **7 Veto Rules**:
  1. RL SHORT + Heatmap Heavy Support → VETO
  2. LLM `veto_recommendation=True` → VETO
  3. All agents confidence < 0.50 → VETO
  4. EXTREME_LONG funding for new LONG → VETO
  5. EXTREME_SHORT funding for new SHORT → VETO
  6. Consensus score < threshold → VETO
  7. No directional consensus → VETO

## Validation Results (10,000-bar Backtest)

| Metric | Value | Target |
|--------|-------|--------|
| Bars Processed | 10,000 | 10,000 |
| Signals Generated | 934 | — |
| Signals Vetoed | 1,850 | — |
| Total Trades | 568 | — |
| Win Rate | 29.75% | — |
| Profit Factor | 0.773 | >1.8 |
| Head Agent Veto Rate | 66.5% | — |

> **Note**: PF<1.8 is expected with untrained PatchTST/PPO models. Train models on historical data or FinRL environment to achieve target. The architecture and selectivity mechanics are fully operational.

## Fractional Differentiation

The `FractionalDifferentiator` implements AFML Chapter 5:
- **Method**: Fixed-width window (FFD) — computationally efficient for streaming
- **Default d=0.4**: Balances stationarity and memory retention
- **MinD finder**: `FractionalDifferentiator.find_min_d(prices)` via ADF test grid search

## Data Bus

The `MarketDataBus` supports two backends:
- **InProcessBus** (default): Pure asyncio, no external deps
- **RedisBus**: `pip install redis[hiredis]` + `bus.redis_enabled: true`

Channels: `ORDERBOOK`, `HEATMAP`, `CONTEXT`, `TRADE`, `SIGNAL`, `HEARTBEAT`
