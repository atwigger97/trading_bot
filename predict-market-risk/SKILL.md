---
name: predict-market-risk
description: |
  Autonomous Polymarket prediction market trading bot. Runs a 6-agent pipeline:
  research (sentiment), filter (opportunity scoring), predict (XGBoost + Claude calibration),
  risk (Kelly sizing), execute (CLOB orders), and learn (post-mortem). Use this skill when
  the user says: "run the trading bot", "check prediction markets", "what markets should I trade",
  "scan polymarket", "run market analysis", "check my open positions", "show trading performance",
  "retrain the model", "start dry run", "pause the bot", or any question about Polymarket trading,
  risk parameters, P&L, open positions, predictions, or XGBoost model status.
metadata:
  author: alext
  version: 1.0.0
  category: trading
---

## Overview

This bot is an autonomous multi-agent swarm that trades on Polymarket — a prediction market platform where users buy YES/NO shares on real-world events. The pipeline runs every 5 minutes: it ingests live markets from the Polymarket CLOB and Gamma APIs, researches sentiment across Twitter, Reddit, and RSS feeds (scored via VADER + FinBERT), filters for high-opportunity markets by volume/liquidity/time, generates a calibrated probability using XGBoost plus Claude Sonnet calibration, sizes bets using fractional Kelly Criterion (0.25x) with hard risk caps, and places limit orders via the py-clob-client SDK on Polygon mainnet. After trades settle, a learning agent calls Claude to perform post-mortem analysis on losses and flags feature weight adjustments for model retraining. All state is persisted in a SQLite database with WAL mode for concurrent reads.

## Quick Commands

| What to say | What it runs |
|---|---|
| "start the bot" / "go live" | `python scripts/run_bot.py loop` |
| "dry run" / "test run" | `python scripts/run_bot.py dry-run` |
| "ingest markets" / "refresh markets" | `python scripts/run_bot.py ingest` |
| "train the model" / "retrain XGBoost" | `python models/xgboost_model.py train` |
| "check model status" | `python models/xgboost_model.py info` |
| "show my P&L" / "how am I doing" | See Monitoring Queries — Today's P&L |
| "show open positions" | See Monitoring Queries — Open Positions |
| "show recent predictions" | See Monitoring Queries — Recent Predictions |
| "pause the bot" / "stop trading" | Kill the running `run_bot.py` process |
| "show learnings" / "what went wrong" | See Monitoring Queries — Learnings |

When the user says "pause the bot", find and terminate the running `python scripts/run_bot.py` process. On Windows use `Get-Process python | Stop-Process`, on Linux/macOS use `pkill -f run_bot`.

## Risk Parameters (Current Config)

These values live in `config.py` under `RiskConfig`. Report them when the user asks about risk settings or position limits.

| Parameter | Value | Meaning |
|---|---|---|
| `kelly_fraction` | 0.25 | Quarter-Kelly — conservative sizing |
| `max_kelly_bet_pct` | 0.05 (5%) | Hard cap: never risk more than 5% of bankroll per trade |
| `max_single_position_usdc` | $500 | Maximum dollars in any single position |
| `max_total_exposure_usdc` | $2,000 | Maximum total open exposure across all positions |
| `daily_loss_limit_usdc` | $300 | Bot pauses if daily realized loss exceeds this |
| `min_edge_pct` | 0.04 (4%) | Minimum edge (our_prob - market_price) to place a trade |
| `min_liquidity_usdc` | $1,000 | Skip markets with order book depth below this |
| `min_volume_24h_usdc` | $500 | Skip markets with 24h volume below this |
| `max_days_to_resolution` | 30 | Skip markets resolving more than 30 days out |
| `sentiment_weight` | 0.15 | How much sentiment shifts the heuristic prediction |
| `bankroll_usdc` | $1,000 | Starting bankroll for Kelly sizing |

## Pipeline Reference

### 1. Research Agent (`agents/research_agent.py`)
**Function:** `get_sentiment(market)` — scrapes Twitter (API v2), Reddit (PRAW), and RSS (feedparser) for content matching the market question keywords. Scores each text with VADER + FinBERT ensemble. Returns `{"twitter": float, "reddit": float, "rss": float, "composite": float}` in [-1, +1]. Persists per-source scores to the `sentiment` table. Gracefully skips any source whose API key is missing or rate-limited.

### 2. Filter Agent (`agents/filter_agent.py`)
**Function:** `rank_markets(top_n)` — pulls active markets from DB, scores each on volume (40%), liquidity (30%), and time-to-resolution (30%), plus a sentiment strength bonus. Returns top N candidates sorted by opportunity score. Pre-filters by the same thresholds as ingestion (liquidity, price bounds, days).

### 3. Predict Agent (`agents/predict_agent.py`)
**Function:** `predict(market, sentiment)` — builds a feature vector, runs XGBoost (or heuristic baseline if no model file exists), then calls Claude Sonnet to calibrate the probability. Claude returns a JSON with `claude_prob`, `confidence`, and `reasoning`. Computes edge and direction (YES/NO). Skips markets below `min_edge_pct`. Persists to `predictions` table.

### 4. Risk Agent (`agents/risk_agent.py`)
**Function:** `approve_trade(prediction, market)` — checks daily loss limit, sizes the bet via fractional Kelly, caps at config limits, and checks total exposure. Returns `(approved: bool, size_usdc: float)`. If exposure is tight, reduces size to fit rather than rejecting outright.

### 5. Execute Agent (`agents/execute_agent.py`)
**Function:** `place_order(market, prediction, size_usdc)` — authenticates with the Polymarket CLOB via `py-clob-client`, places a BUY limit order at the current best price, persists to the `trades` table, and polls for fill status (up to 10 polls at 3-second intervals). Returns the order ID or None.

### 6. Learn Agent (`agents/learn_agent.py`)
**Function:** `review_settled_trades()` — queries trades that are settled but have no learning entry. For wins, logs a summary. For losses, calls Claude Sonnet with full trade context to perform post-mortem analysis. Persists `error_analysis` and `feature_flags` (JSON with suggested feature weight changes) to the `learnings` table for future XGBoost retraining.

## Monitoring Queries

Run these Python snippets inline when the user asks for status information.

### Today's P&L

```python
from data.db import get_daily_pnl
pnl = get_daily_pnl()
print(f"Today's realized P&L: ${pnl:+.2f} USDC")
```

### Open Positions

```python
from data.db import get_open_trades, get_open_exposure
trades = get_open_trades()
exposure = get_open_exposure()
for t in trades:
    print(f"  {t['condition_id'][:8]} | {t['direction']} | ${t['size_usdc']:.2f} | {t['status']}")
print(f"Total open exposure: ${exposure:.2f}")
```

### Last 10 Trades with Outcome

```python
from data.db import get_conn
with get_conn() as conn:
    rows = conn.execute("""
        SELECT condition_id, direction, size_usdc, avg_fill_price,
               status, pnl_usdc, placed_at
        FROM trades ORDER BY placed_at DESC LIMIT 10
    """).fetchall()
for r in rows:
    pnl = f"${r['pnl_usdc']:+.2f}" if r['pnl_usdc'] else "pending"
    print(f"  {r['condition_id'][:8]} | {r['direction']} ${r['size_usdc']:.2f} "
          f"@ {r['avg_fill_price'] or '?'} | {r['status']} | {pnl}")
```

### Recent Predictions with Edge

```python
from data.db import get_conn
with get_conn() as conn:
    rows = conn.execute("""
        SELECT p.condition_id, m.question, p.claude_prob, p.market_yes_price,
               p.edge_pct, p.confidence, p.predicted_at
        FROM predictions p
        JOIN markets m ON p.condition_id = m.condition_id
        ORDER BY p.predicted_at DESC LIMIT 10
    """).fetchall()
for r in rows:
    print(f"  {r['question'][:50]} | claude={r['claude_prob']:.2f} "
          f"mkt={r['market_yes_price']:.2f} | edge={r['edge_pct']:+.2%} "
          f"| {r['confidence']}")
```

### Learnings from Losses

```python
from data.db import get_conn
with get_conn() as conn:
    rows = conn.execute("""
        SELECT l.outcome, l.pnl_usdc, l.error_analysis, l.feature_flags,
               m.question
        FROM learnings l
        JOIN markets m ON l.condition_id = m.condition_id
        WHERE l.outcome = 'loss'
        ORDER BY l.created_at DESC LIMIT 5
    """).fetchall()
for r in rows:
    print(f"  LOSS ${r['pnl_usdc']:+.2f} — {r['question'][:50]}")
    print(f"    Analysis: {r['error_analysis']}")
    print(f"    Flags: {r['feature_flags']}")
```

## Troubleshooting

### Bot runs but places no trades
**Cause:** Edge threshold is too high relative to market efficiency, or all markets are filtered out.
**Fix:** Check how many candidates pass the filter. Run `python scripts/run_bot.py dry-run` and watch the logs. If you see "Insufficient edge" on every market, consider temporarily lowering `min_edge_pct` from 0.04 to 0.03 in `config.py`. Also verify that `ingest` saved enough markets — if the DB has zero active markets, ingestion may have failed.

### Orders stuck in pending status
**Cause:** Limit price was too aggressive (below best bid) or Polymarket order book is thin.
**Fix:** Check the CLOB API status at `https://clob.polymarket.com/markets`. Review the trade log for the order ID and query it directly. If orders consistently hang, the bot's price may be stale — ensure `refresh_prices()` is running. You can manually cancel pending orders through the Polymarket UI.

### XGBoost training fails (not enough resolved markets)
**Cause:** The Gamma API returned fewer than 200 resolved markets with sufficient volume.
**Fix:** This is normal early on. The bot uses a heuristic fallback (market price adjusted by sentiment) until enough data accumulates. Wait for more markets to resolve, then re-run `python models/xgboost_model.py train`. Lower `min_samples` in the train() call as a last resort, but models under 150 samples are unreliable.

### Research agent skips all sentiment sources
**Cause:** Missing or invalid API keys for Twitter, Reddit, or RSS feeds unreachable.
**Fix:** Check `.env` for `TWITTER_BEARER_TOKEN`, `REDDIT_CLIENT_ID`, and `REDDIT_CLIENT_SECRET`. The bot will still work without them — sentiment defaults to 0.0 (neutral) — but predictions lose an input signal. RSS requires no credentials and should always work unless the feeds are down. Check logs for specific "skipping" messages.

### Daily loss limit triggered
**Cause:** Cumulative P&L for the day hit -$300 (or whatever `daily_loss_limit_usdc` is set to).
**Fix:** This is a safety feature. The bot will resume trading the next calendar day. If you want to override, update `daily_loss_limit_usdc` in `config.py`. Do not remove the limit entirely — it prevents catastrophic drawdown. Review the learnings table to understand what caused the losses before resuming.

## Retraining Schedule

**When to retrain:** Weekly, or after 50+ new markets have resolved since the last training run. The model needs resolved markets with known outcomes as training labels.

**Command:**
```
python models/xgboost_model.py train
```

**What to aim for:**
- ROC-AUC above **0.65** — the model has genuine predictive signal and is safe to trade on
- ROC-AUC 0.55–0.65 — marginal signal, use with caution, rely more on Claude calibration
- ROC-AUC near 0.50 — no better than random; stay in heuristic mode until more data arrives

**After training:** The model is auto-saved to `models/xgboost_model.pkl`. The predict agent will load it on next run — no restart needed. Check metrics with `python models/xgboost_model.py info`.

**Brier score** should be below 0.25 (lower is better). This measures calibration quality — how close predicted probabilities are to actual outcomes. The Platt scaling calibration layer in training helps with this.
