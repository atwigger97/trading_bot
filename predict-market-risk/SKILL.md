---
name: predict-market-risk
description: |
  Autonomous Polymarket prediction market trading bot running on VPS 108.160.141.236. Manages a
  7-agent pipeline (research, filter, predict, risk, execute, learn, notify) that trades YES/NO
  markets every 5 minutes on Polygon mainnet. Focuses on short-duration (1-3 day) markets in
  sports, crypto, economics, and politics. Use this skill for: "check bot status", "show positions",
  "show P&L", "pause the bot", "restart the bot", "train the model", "check open trades",
  "how is the bot doing", "run the trading bot", "check prediction markets", "scan polymarket",
  "run market analysis", "show trading performance", "retrain the model", "start dry run",
  or any question about Polymarket trading, risk parameters, open positions, predictions,
  XGBoost model status, or P&L.
metadata:
  author: alext
  version: 2.0.0
  category: trading
---

## Overview

This bot is an autonomous multi-agent swarm that trades on Polymarket — a prediction market platform where users buy YES/NO shares on real-world events. It runs on VPS **108.160.141.236** as a systemd service (`prediction-bot`). The pipeline runs every 5 minutes: it ingests live markets from the Polymarket CLOB and Gamma APIs, researches sentiment across Google News, Reddit, and RSS feeds (scored via VADER + FinBERT), filters for short-duration (1-3 day) markets in high-accuracy categories (sports, crypto, economics, politics), generates a calibrated probability using XGBoost plus Claude Sonnet calibration, sizes bets using fractional Kelly Criterion (0.25x) with hard risk caps, and places limit orders via the py-clob-client SDK on Polygon mainnet. After trades settle, a learning agent calls Claude to perform post-mortem analysis on losses and flags feature weight adjustments for model retraining. Notifications are sent via CallMeBot WhatsApp. All state is persisted in a SQLite database with WAL mode for concurrent reads.

**Strategy Focus (as of March 2026):** Short-duration markets only (≤3 days to resolution). Category priority: sports (1.3×), crypto/economics (1.2×), politics (0.9×). Entertainment and science categories are blocked. Minimum $2,000 liquidity, minimum $1,000 24h volume. Spread check: yes+no must be between 0.90–1.10. Price bounds: 0.15–0.85 (no near-certain markets).

## Quick Commands

All commands run on the VPS. SSH first if accessing remotely:

```
ssh root@108.160.141.236
```

### Check status / P&L / open positions

```bash
cd /root/prediction-bot/trading_bot
source venv/bin/activate
PYTHONPATH=/root/prediction-bot/trading_bot python3 scripts/status.py
```

### Check logs

```bash
tail -50 /root/prediction-bot/trading_bot/logs/bot.log
```

### Restart bot

```bash
systemctl restart prediction-bot
```

### Pause bot

```bash
systemctl stop prediction-bot
```

### Start bot

```bash
systemctl start prediction-bot
```

### Show service status

```bash
systemctl status prediction-bot
```

### Train model

```bash
cd /root/prediction-bot/trading_bot
source venv/bin/activate
PYTHONPATH=/root/prediction-bot/trading_bot python3 models/xgboost_model.py train
```

### Check open positions (inline Python)

```bash
cd /root/prediction-bot/trading_bot && source venv/bin/activate
PYTHONPATH=/root/prediction-bot/trading_bot python3 - <<'EOF'
from data.db import get_open_trades, get_open_exposure
for t in get_open_trades():
    print(f"{t['condition_id'][:8]} | {t['direction']} | ${t['size_usdc']:.2f} | {t['status']}")
print(f"Total exposure: ${get_open_exposure():.2f}")
EOF
```

| Say | Action |
|---|---|
| "start the bot" / "go live" | `systemctl start prediction-bot` |
| "dry run" / "test run" | `python scripts/run_bot.py dry-run` |
| "ingest markets" | `python scripts/run_bot.py ingest` |
| "train the model" | `python models/xgboost_model.py train` |
| "pause the bot" | `systemctl stop prediction-bot` |
| "restart the bot" | `systemctl restart prediction-bot` |
| "check bot status" | `systemctl status prediction-bot` |
| "show positions" / "check open trades" | inline Python snippet above |
| "check logs" | `tail -50 logs/bot.log` |

When the user says "pause the bot", run `systemctl stop prediction-bot`. On systems without systemd, use `pkill -f run_bot`.

## Systemd Service

The bot runs as a systemd service at `/etc/systemd/system/prediction-bot.service`. It is enabled to start on boot.

| Command | What it does |
|---|---|
| `systemctl start prediction-bot` | Start the bot |
| `systemctl stop prediction-bot` | Stop the bot |
| `systemctl restart prediction-bot` | Restart after config changes |
| `systemctl status prediction-bot` | Check if running |
| `journalctl -u prediction-bot -f` | Follow live logs |

### Switch to live mode

The service defaults to `dry-run`. To go live, edit the service file:
```bash
sudo sed -i 's/dry-run/loop/' /etc/systemd/system/prediction-bot.service
sudo systemctl daemon-reload && sudo systemctl restart prediction-bot
```

To switch back to dry-run:
```bash
sudo sed -i 's/loop/dry-run/' /etc/systemd/system/prediction-bot.service
sudo systemctl daemon-reload && sudo systemctl restart prediction-bot
```

## Infrastructure

| Item | Value |
|---|---|
| VPS IP | 108.160.141.236 |
| OS | Linux (Ubuntu) |
| Python | 3.12 (venv at `/root/prediction-bot/trading_bot/venv/`) |
| Database | SQLite WAL at `data/prediction_bot.db` |
| Model | XGBoost at `models/xgboost_model.pkl` |
| Wallet | `0x69FDC4f78B3444a58358BEE267235353BFe780a9` |
| Chain | Polygon mainnet (chain ID 137) |
| USDC.e contract | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` |
| RPC | `polygon-bor-rpc.publicnode.com` (`polygon-rpc.com` returns 401) |
| Claude model | `claude-sonnet-4-6` |
| Notifications | CallMeBot WhatsApp (env: `WHATSAPP_PHONE`, `WHATSAPP_APIKEY`) |

## Risk Parameters (Current Config)

These values live in `config.py` under `RiskConfig`. Report them when the user asks about risk settings or position limits.

| Parameter | Value | Meaning |
|---|---|---|
| `bankroll_usdc` | ~$105 (live on-chain) | Fetched from Polygon USDC.e balance + open exposure, cached 4.5 min |
| `kelly_fraction` | 0.25 | Quarter-Kelly — conservative sizing |
| `max_kelly_bet_pct` | 0.10 (10%) | Hard cap: never risk more than 10% of bankroll per trade |
| `max_single_position_usdc` | $15 | Maximum dollars in any single position |
| `max_total_exposure_usdc` | $100 | Maximum total open exposure across all positions |
| `daily_loss_limit_usdc` | $25 | Bot pauses if daily realized loss exceeds this |
| `min_edge_pct` | 0.06 (6%) | Minimum edge (our_prob − market_price) to place a trade |
| `min_liquidity_usdc` | $2,000 | Skip markets with order book depth below this |
| `min_volume_24h_usdc` | $1,000 | Skip markets with 24h volume below this |
| `max_days_to_resolution` | 3 | Only trade markets resolving within 3 days |
| `long_threshold_days` | 3 | Markets beyond this count against the long-exposure cap |
| `max_long_exposure_pct` | 40% | Cap on exposure in "long" markets (>3 days) |
| `sentiment_weight` | 0.15 | How much sentiment shifts the heuristic prediction |

### Category Multipliers (filter_agent.py)

Applied to the final opportunity score to prioritise high-accuracy categories:

| Category | Multiplier |
|---|---|
| sports | 1.3× |
| crypto | 1.2× |
| economics | 1.2× |
| finance | 1.0× |
| politics | 0.9× |
| science | 0.7× |
| technology | 0.7× |
| entertainment | 0.5× (also hard-blocked in prefilter) |
| (unknown) | 0.7× |

### Prefilter Rules (filter_agent.py)

A market is **rejected before scoring** if any of these are true:
- `active != 1` or `closed == 1`
- liquidity < $2,000
- yes_price > 0.85 or yes_price < 0.15 (no near-certain markets)
- yes_price + no_price < 0.90 or > 1.10 (broken spread)
- days_to_resolution is None, ≤ 0, or > 3
- category is "entertainment"

## Pipeline Reference

### 1. Research Agent (`agents/research_agent.py`)
**Function:** `get_sentiment(market)` — fetches articles from Google News RSS search, Reddit (PRAW), and 9 static RSS feeds (BBC, NYT, Reuters, Politico, CryptoNews, NPR, Al Jazeera, etc.) for content matching the market question keywords. Scores each text with VADER + FinBERT ensemble. Returns `{"news": float, "reddit": float, "rss": float, "composite": float}` in [-1, +1]. Weights: news 45%, reddit 30%, rss 25%. Persists per-source scores to the `sentiment` table. Gracefully skips any source that fails or has no credentials.

### 2. Filter Agent (`agents/filter_agent.py`)
**Function:** `rank_markets(top_n)` — pulls active markets from DB, applies prefilter (see Prefilter Rules above), then scores each on volume (30%), liquidity (25%), and time-to-resolution (45%). Applies a time-horizon multiplier (same-day: 2×, short 1–3 days: 1.5×, medium: 1×, long: 0.5×) and a category multiplier (see Category Multipliers above). Returns top N candidates sorted by composite opportunity score. Only considers markets resolving within 3 days with at least $2,000 liquidity and $1,000 24h volume.

### 3. Predict Agent (`agents/predict_agent.py`)
**Function:** `predict(market, sentiment)` — builds a feature vector, runs XGBoost (or heuristic baseline if no model file exists), then calls Claude Sonnet to calibrate the probability. Claude is given a category-specific system prompt (sports, crypto, economics, politics, entertainment each have tailored prompts). Claude returns a JSON with `claude_prob`, `confidence`, and `reasoning`. Computes edge and direction (YES/NO). Skips markets below `min_edge_pct`. Rejects predictions with `confidence == "low"`. Persists to `predictions` table.

### 4. Risk Agent (`agents/risk_agent.py`)
**Function:** `approve_trade(prediction, market)` — fetches live bankroll from on-chain USDC.e balance (cached 4.5 min, RPC: `polygon-bor-rpc.publicnode.com`), checks daily loss limit, sizes the bet via fractional Kelly, caps at config limits, checks total exposure against $100 cap, enforces the long-exposure cap (max 40% of total in markets >3 days out), and runs a correlation check via Claude when 3+ positions are open (detects conflicting directional bets on related events). Returns `(approved: bool, size_usdc: float)`. Reduces size to fit remaining capacity rather than outright rejecting if exposure is close to the cap.

### 5. Execute Agent (`agents/execute_agent.py`)
**Function:** `place_order(market, prediction, size_usdc)` — authenticates with the Polymarket CLOB via `py-clob-client`, places a BUY limit order at the current best price, persists to the `trades` table, and polls for fill status (up to 10 polls at 3-second intervals). Returns the order ID or None.

**`reconcile_pending_orders()`** — called at the top of every trading cycle. Re-checks all `pending` trades against the CLOB API. Handles three cases: (1) SDK raises `AttributeError` — order is archived/settled on CLOB, marks DB record as `filled`; (2) API returns `None` — same treatment; (3) API returns a status — updates DB to `filled`, `cancelled`, or `expired` accordingly.

### 6. Learn Agent (`agents/learn_agent.py`)
**Function:** `review_settled_trades()` — queries trades that are settled but have no learning entry. For wins, logs a summary. For losses, calls Claude Sonnet with full trade context to perform post-mortem analysis. Persists `error_analysis` and `feature_flags` (JSON with suggested feature weight changes) to the `learnings` table for future XGBoost retraining. Runs daily at 00:00 via scheduler in `run_bot.py`.

### 7. Notify Agent (`agents/notify_agent.py`)
**Function:** `send_notification(message)` — sends a WhatsApp message via CallMeBot free API (GET request, no server needed, 300 msg/day limit). No HTML tags — plain text only.

Helper functions: `notify_trade_placed(market, prediction, size_usdc)`, `notify_trade_filled(trade)`, `notify_daily_loss_limit(daily_pnl)`, `notify_daily_summary(stats)`.

**Required env vars:** `WHATSAPP_PHONE` (international format, no `+`), `WHATSAPP_APIKEY` (from CallMeBot — send "I allow callmebot to send me messages" to +34 644 58 62 95 on WhatsApp first). If vars are missing, notifications are silently skipped.

## Scheduled Jobs (run_bot.py)

| Time | Job |
|---|---|
| Every 5 min | Full trading cycle (ingest → research → filter → predict → risk → execute) |
| 00:00 daily | `learn_agent.review_settled_trades()` + daily summary WhatsApp |
| Sunday 02:00 | `retrain_model()` — XGBoost retrain from `files/xgboost_model.py` |

## Trade Flow (run_bot.py trading_cycle)

1. `reconcile_pending_orders()` — update status of any pending trades
2. Check total exposure — if ≥ $100, skip the rest of this cycle
3. `ingest_markets()` — scan 300 markets from Polymarket CLOB/Gamma
4. `get_open_trades()` — build de-dup set of condition_ids already held
5. `rank_markets(top_n=5)` — score and filter candidates (max 5 evaluated per cycle)
6. For each candidate (max 2 new trades per cycle):
   - Skip if already in open positions (de-dup)
   - `get_sentiment()` → `predict()` → reject if confidence is "low"
   - `approve_trade()` → place order if approved
7. Sleep 5 minutes

## Contracts (all have MAX USDC.e allowance already approved)

| Contract | Address |
|---|---|
| CTF Exchange | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` |
| Neg Risk Exchange | `0xC5d563A36AE78145C45a50134d48A1215220f80a` |
| Neg Risk Adapter | `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` |

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

### Performance Summary

```python
from data.db import get_performance_stats
s = get_performance_stats()
print(f"Total trades: {s['total_trades']} | Win rate: {s['win_rate']:.1%}")
print(f"Total P&L: ${s['total_pnl']:+.2f} | Avg: ${s['avg_pnl']:+.2f}")
print(f"Best: ${s['best_trade']:+.2f} | Worst: ${s['worst_trade']:+.2f}")
for cat, stats in s.get('by_category', {}).items():
    print(f"  {cat}: {stats['count']} trades, ${stats['pnl']:+.2f}")
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

### Bot stopped trading (hit exposure cap)
This is normal. Once total open exposure reaches $100, the bot skips new trades until positions settle. Check exposure with `status.py` or the open positions snippet.

### Bot runs but places no new trades (below cap)
**Cause:** Edge threshold (6%) filtering everything out, all good markets already have open positions (de-dup), per-cycle limit (2 trades/cycle) reached, or markets available don't pass the 3-day duration filter.

**Fix:** Check logs for "Insufficient edge", "Low confidence", "After de-dup: 0", or "Max exposure reached" messages. Run `python scripts/run_bot.py dry-run` to test without placing orders. If you see "5/300 markets passed filter" or similar, the filter is very tight by design — short-duration markets are less common.

### Reconcile warnings: "CLOB returned null"
This is now handled gracefully. Old trades (from before the `reconcile` fix) that the CLOB API can no longer find are automatically marked as `filled` in the DB. This clears them from the "pending" state and frees exposure. You will see a log line: `[reconcile] Trade #N: CLOB returned null (archived/settled) → marked filled`. No action needed.

### Orders stuck in pending status
**Cause:** Limit price was too aggressive (below best bid) or Polymarket order book is thin.
**Fix:** Check the CLOB API status at `https://clob.polymarket.com/markets`. Review the trade log for the order ID and query it directly. If orders consistently hang, the bot's price may be stale — ensure `refresh_prices()` is running. You can manually cancel pending orders through the Polymarket UI.

### XGBoost training fails (not enough resolved markets)
**Cause:** The Gamma API returned fewer than 200 resolved markets with sufficient volume.
**Fix:** This is normal early on. The bot uses a heuristic fallback (market price adjusted by sentiment) until enough data accumulates. Wait for more markets to resolve, then re-run `python models/xgboost_model.py train`. Lower `min_samples` in the train() call as a last resort, but models under 150 samples are unreliable.

### Research agent returns no sentiment data
**Cause:** Google News rate-limited, Reddit credentials missing, or all RSS feeds down.
**Fix:** Google News RSS and the 9 static feeds require no credentials and should always work. Check `.env` for `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` if Reddit scores are always 0. The bot will still trade without sentiment — it defaults to 0.0 (neutral) and relies on XGBoost + Claude. Check logs for "Google News search failed" or "RSS error" messages.

### Daily loss limit triggered
**Cause:** Cumulative P&L for the day hit -$25 (`daily_loss_limit_usdc`).
**Fix:** This is a safety feature. The bot will resume trading the next calendar day. If you want to override, update `daily_loss_limit_usdc` in `config.py`. Do not remove the limit. Review the learnings table to understand what caused the losses before resuming.

### WhatsApp notifications not working
**Cause:** `WHATSAPP_PHONE` or `WHATSAPP_APIKEY` not set in `.env`, or CallMeBot API key not activated.
**Fix:** 
1. Send "I allow callmebot to send me messages" to +34 644 58 62 95 on WhatsApp
2. You'll receive your API key in reply
3. Add to `.env`: `WHATSAPP_PHONE=447XXXXXXXXX` (no `+`) and `WHATSAPP_APIKEY=your_key`
4. Restart the bot

### Wrong RPC endpoint
`polygon-rpc.com` returns 401 for live balance queries. Always use `polygon-bor-rpc.publicnode.com`.

## Retraining Schedule

**When to retrain:** Weekly (auto-scheduled Sunday 02:00), or after 50+ new markets have resolved since the last training run.

**Command:**
```
python models/xgboost_model.py train
```

**What to aim for:**
- ROC-AUC above **0.65** — the model has genuine predictive signal and is safe to trade on
- ROC-AUC 0.55–0.65 — marginal signal, use with caution, rely more on Claude calibration
- ROC-AUC near 0.50 — no better than random; stay in heuristic mode until more data arrives

**After training:** The model is auto-saved to `models/xgboost_model.pkl`. The predict agent will load it on next run — no restart needed. Check metrics with `python models/xgboost_model.py info`.

**Brier score** should be below 0.25 (lower is better). This measures calibration quality — how close predicted probabilities are to actual outcomes.

