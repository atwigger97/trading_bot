"""
agents/learn_agent.py — Post-mortem analysis on settled trades.

For each settled trade without a learning entry:
  1. Gather full context (market, prediction, trade outcome)
  2. For losses: call Claude Sonnet for post-mortem analysis
  3. Persist to learnings table
  4. Flag feature importance adjustments for XGBoost retraining

Public API (called by run_bot.trading_cycle):
    review_settled_trades() → int  (number of trades reviewed)
"""

import json
import logging
from typing import Optional

import anthropic

from config import CLAUDE_MODEL, ANTHROPIC_API_KEY
from data.db import (
    get_unreviewed_settled_trades,
    get_market,
    get_latest_prediction,
    save_learning,
)
from data.normalizer import format_market_for_claude

logger = logging.getLogger(__name__)


def _build_context(trade: dict) -> Optional[dict]:
    """Assemble full context for post-mortem: market + prediction + outcome."""
    condition_id = trade["condition_id"]

    market = get_market(condition_id)
    if not market:
        logger.warning(f"[trade #{trade['id']}] Market {condition_id[:8]} not found")
        return None

    prediction = get_latest_prediction(condition_id)

    pnl = trade.get("pnl_usdc") or 0
    outcome = "win" if pnl > 0 else "loss" if pnl < 0 else "break-even"

    return {
        "trade_id":         trade["id"],
        "condition_id":     condition_id,
        "outcome":          outcome,
        "pnl_usdc":         pnl,
        "direction":        trade.get("direction", "?"),
        "size_usdc":        trade.get("size_usdc", 0),
        "avg_fill_price":   trade.get("avg_fill_price"),
        "question":         market.get("question", ""),
        "category":         market.get("category", ""),
        "yes_price":        market.get("yes_price"),
        "our_prob":         prediction.get("claude_prob") if prediction else None,
        "market_prob":      prediction.get("market_yes_price") if prediction else None,
        "edge_pct":         prediction.get("edge_pct") if prediction else None,
        "confidence":       prediction.get("confidence") if prediction else None,
        "claude_reasoning": prediction.get("claude_reasoning") if prediction else None,
        "market":           market,
    }


def _claude_postmortem(ctx: dict) -> dict:
    """
    Call Claude to analyze a losing / break-even trade.

    Returns {"error_analysis": str, "feature_flags": str (JSON)}.
    """
    if not ANTHROPIC_API_KEY:
        return {
            "error_analysis": "Post-mortem skipped (no API key)",
            "feature_flags": "{}",
        }

    prompt = f"""You are a prediction market analyst reviewing a completed trade.

TRADE:
- Market: {ctx['question']}
- Category: {ctx['category']}
- Direction: {ctx['direction']}
- Fill price: {ctx['avg_fill_price']}
- Size: ${ctx['size_usdc']}
- P&L: ${ctx['pnl_usdc']:+.2f} ({ctx['outcome']})

PREDICTION AT TIME OF TRADE:
- Our probability: {ctx['our_prob']}
- Market price:    {ctx['market_prob']}
- Edge:            {ctx['edge_pct']}
- Confidence:      {ctx['confidence']}
- Reasoning:       {ctx['claude_reasoning']}

TASK:
1. What signal or assumption was incorrect?
2. Was sentiment, model, or calibration the weakest link?
3. Suggest feature weight adjustments as a JSON object.

Respond ONLY with valid JSON:
{{"error_analysis": "2-3 sentence analysis", "feature_flags": {{"sentiment_weight": "...", "category_note": "..."}}}}"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        result = json.loads(raw)
        flags = result.get("feature_flags", {})
        if isinstance(flags, dict):
            flags = json.dumps(flags)
        return {
            "error_analysis": result.get("error_analysis", ""),
            "feature_flags": flags,
        }
    except json.JSONDecodeError:
        logger.warning("Claude post-mortem returned invalid JSON")
    except anthropic.RateLimitError:
        logger.warning("Claude rate limited during post-mortem")
    except Exception as e:
        logger.error(f"Claude post-mortem failed: {e}")

    return {"error_analysis": "Post-mortem failed", "feature_flags": "{}"}


def _review_one(trade: dict) -> bool:
    """Perform post-mortem on a single settled trade. Returns True on success."""
    ctx = _build_context(trade)
    if ctx is None:
        return False

    logger.info(
        f"[trade #{trade['id']}] Reviewing {ctx['outcome']} "
        f"(${ctx['pnl_usdc']:+.2f}) — {ctx['question'][:60]}"
    )

    # Only call Claude for losses / break-even (save API costs on wins)
    if ctx["outcome"] == "win":
        analysis = {
            "error_analysis": (
                f"Win: {ctx['direction']} trade profitable (${ctx['pnl_usdc']:+.2f}). "
                "Signals aligned correctly."
            ),
            "feature_flags": "{}",
        }
    else:
        analysis = _claude_postmortem(ctx)

    # Determine resolution from P&L + direction
    if ctx["pnl_usdc"] > 0:
        resolution = ctx["direction"]               # we were right
    elif ctx["pnl_usdc"] < 0:
        resolution = "NO" if ctx["direction"] == "YES" else "YES"
    else:
        resolution = "UNKNOWN"

    learning_row = {
        "trade_id":       trade["id"],
        "condition_id":   ctx["condition_id"],
        "outcome":        ctx["outcome"],
        "pnl_usdc":       ctx["pnl_usdc"],
        "our_prob":       ctx["our_prob"],
        "market_prob":    ctx["market_prob"],
        "resolution":     resolution,
        "error_analysis": analysis["error_analysis"],
        "feature_flags":  analysis["feature_flags"],
    }

    try:
        save_learning(learning_row)
        logger.info(f"[trade #{trade['id']}] Learning saved")
        return True
    except Exception as e:
        logger.error(f"[trade #{trade['id']}] Failed to save learning: {e}")
        return False


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def review_settled_trades() -> int:
    """
    Review all settled trades that haven't been analyzed yet.

    Returns number of trades successfully reviewed.
    """
    trades = get_unreviewed_settled_trades()
    if not trades:
        logger.debug("Learn agent: no unreviewed trades")
        return 0

    logger.info(f"Learn agent: {len(trades)} trades to review")

    reviewed = 0
    for trade in trades:
        try:
            if _review_one(trade):
                reviewed += 1
        except Exception as e:
            logger.error(f"[trade #{trade['id']}] Review failed: {e}")

    logger.info(f"Learn agent: {reviewed}/{len(trades)} trades reviewed")
    return reviewed
