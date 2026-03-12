"""
agents/predict_agent.py — Probability prediction via XGBoost + Claude calibration.

Pipeline:
  1. Build feature vector from market data + sentiment
  2. XGBoost prediction (or heuristic baseline if no model file)
  3. Claude Sonnet calibration: validates / adjusts probability
  4. Persist prediction to DB via save_prediction()

Public API (called by run_bot._process_market):
    predict(market, sentiment) → dict | None

Returns:
    {"xgboost_prob": float, "sentiment_adj_prob": float,
     "claude_prob": float, "direction": "YES"|"NO",
     "edge_pct": float, "confidence": str, "claude_reasoning": str}
    or None if no tradeable edge.
"""

import os
import json
import logging
import pickle
from typing import Optional

import anthropic

from config import CLAUDE_MODEL, ANTHROPIC_API_KEY, RISK
from data.db import save_prediction
from data.normalizer import edge, format_market_for_claude

logger = logging.getLogger(__name__)

# ─── XGBOOST MODEL (optional) ───────────────────────────────────────────────

_MODEL_PATH = "models/xgboost_model.pkl"
_xgb_model = None


def _load_model():
    """Try to load a trained XGBoost model from disk."""
    global _xgb_model
    if _xgb_model is not None:
        return _xgb_model
    if os.path.exists(_MODEL_PATH):
        try:
            with open(_MODEL_PATH, "rb") as f:
                _xgb_model = pickle.load(f)
            logger.info("XGBoost model loaded")
            return _xgb_model
        except Exception as e:
            logger.warning(f"Failed to load XGBoost model: {e}")
    return None


# ─── CATEGORY ENCODING ──────────────────────────────────────────────────────

_CAT_MAP = {
    "politics": 0, "crypto": 1, "sports": 2, "economics": 3,
    "finance": 3, "entertainment": 4, "science": 5, "technology": 6,
}


def _encode_category(cat: str) -> int:
    """Map category string to integer. Unknown → 99."""
    return _CAT_MAP.get((cat or "").lower().strip(), 99)


# ─── FEATURE BUILDING ───────────────────────────────────────────────────────

def _build_features(market: dict, sentiment: dict) -> dict:
    """
    Build feature dict for prediction model.

    Features: yes_price, volume_24h, liquidity, days_to_resolution,
              sentiment_composite, category_encoded
    """
    return {
        "yes_price":            market.get("yes_price", 0.5),
        "volume_24h":           market.get("volume_24h_usdc", 0),
        "liquidity":            market.get("liquidity_usdc", 0),
        "days_to_resolution":   market.get("days_to_resolution") or 15,
        "sentiment_composite":  sentiment.get("composite", 0.0),
        "category_encoded":     _encode_category(market.get("category", "")),
    }


# ─── HEURISTIC BASELINE ─────────────────────────────────────────────────────

def _heuristic_prob(features: dict) -> float:
    """Baseline when no XGBoost model is available: market price ± sentiment."""
    base = features["yes_price"]
    adj = features["sentiment_composite"] * RISK.sentiment_weight
    return max(0.01, min(0.99, base + adj))


# ─── XGBOOST ────────────────────────────────────────────────────────────────

def _xgb_predict(features: dict) -> Optional[float]:
    """Run XGBoost IF a model file exists. Returns prob or None."""
    model = _load_model()
    if model is None:
        return None
    try:
        import numpy as np
        vec = np.array([[
            features["yes_price"],
            features["volume_24h"],
            features["liquidity"],
            features["days_to_resolution"],
            features["sentiment_composite"],
            features["category_encoded"],
        ]])
        prob = float(model.predict_proba(vec)[0][1])
        return max(0.01, min(0.99, prob))
    except Exception as e:
        logger.warning(f"XGBoost prediction failed: {e}")
        return None


# ─── CLAUDE CALIBRATION ─────────────────────────────────────────────────────

def _claude_calibrate(market: dict, sentiment: dict,
                      model_prob: float) -> dict:
    """
    Call Claude Sonnet to calibrate the model probability.

    Returns {"claude_prob": float, "confidence": str, "reasoning": str}.
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("No Anthropic API key — skipping Claude calibration")
        return {
            "claude_prob": model_prob,
            "confidence": "low",
            "reasoning": "No API key — using model prob unmodified",
        }

    market_block = format_market_for_claude(market, sentiment)

    prompt = f"""You are a prediction market calibration expert. Review the model's
probability estimate and either confirm or adjust it.

{market_block}

MODEL ESTIMATE (YES probability): {model_prob:.4f}

INSTRUCTIONS:
1. Consider base rates, known information, and potential biases.
2. Provide your calibrated probability for YES (0.01 – 0.99).
3. Rate confidence: "high", "medium", or "low".
4. Explain in 2-3 sentences.

Respond ONLY with valid JSON (no markdown):
{{"claude_prob": 0.XX, "confidence": "high|medium|low", "reasoning": "..."}}"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        result = json.loads(raw)
        cp = max(0.01, min(0.99, float(result.get("claude_prob", model_prob))))
        return {
            "claude_prob": cp,
            "confidence": result.get("confidence", "medium"),
            "reasoning": result.get("reasoning", ""),
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Claude returned invalid JSON: {e}")
    except anthropic.RateLimitError:
        logger.warning("Claude rate limited")
    except Exception as e:
        logger.error(f"Claude calibration failed: {e}")

    return {
        "claude_prob": model_prob,
        "confidence": "low",
        "reasoning": "Claude call failed — using model prob",
    }


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def predict(market: dict, sentiment: dict) -> Optional[dict]:
    """
    Generate a calibrated prediction for a market.

    Args:
        market:    Market dict (from DB).
        sentiment: {"twitter": float, "reddit": float, "rss": float,
                     "composite": float} from research_agent.

    Returns dict ready for risk_agent, or None if no tradeable edge.
    """
    condition_id = market["condition_id"]
    yes_price = market.get("yes_price", 0.5)

    # Build features
    features = _build_features(market, sentiment)

    # Model prediction
    xgb_prob = _xgb_predict(features)
    if xgb_prob is not None:
        logger.debug(f"[{condition_id[:8]}] XGBoost prob: {xgb_prob:.4f}")
    else:
        logger.debug(f"[{condition_id[:8]}] Using heuristic baseline")

    # Sentiment-adjusted probability (heuristic if no model)
    sentiment_adj = _heuristic_prob(features)
    base_prob = xgb_prob if xgb_prob is not None else sentiment_adj

    # Claude calibration
    cal = _claude_calibrate(market, sentiment, base_prob)
    claude_prob = cal["claude_prob"]

    # Direction + edge
    if claude_prob >= yes_price:
        direction = "YES"
        edge_pct = edge(claude_prob, yes_price, "YES")
    else:
        direction = "NO"
        edge_pct = edge(claude_prob, yes_price, "NO")

    # Minimum edge gate
    if edge_pct < RISK.min_edge_pct:
        logger.debug(
            f"[{condition_id[:8]}] Edge {edge_pct:.2%} < "
            f"threshold {RISK.min_edge_pct:.0%}, skipping"
        )
        return None

    # Persist
    pred_row = {
        "condition_id":      condition_id,
        "xgboost_prob":      xgb_prob,
        "sentiment_adj_prob": sentiment_adj,
        "claude_prob":       claude_prob,
        "market_yes_price":  yes_price,
        "edge_pct":          round(edge_pct, 4),
        "confidence":        cal["confidence"],
        "claude_reasoning":  cal["reasoning"],
    }
    try:
        pred_id = save_prediction(pred_row)
    except Exception as e:
        logger.error(f"[{condition_id[:8]}] Failed to save prediction: {e}")
        pred_id = None

    result = {
        **pred_row,
        "direction":    direction,
        "prediction_id": pred_id,
    }

    logger.info(
        f"[{condition_id[:8]}] Prediction: {direction} "
        f"xgb={xgb_prob} adj={sentiment_adj:.3f} claude={claude_prob:.3f} "
        f"edge={edge_pct:+.2%} conf={cal['confidence']}"
    )

    return result
