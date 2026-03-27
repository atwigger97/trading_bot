"""
models/xgboost_model.py — XGBoost probability calibration model

Two responsibilities:
  1. train()  — fetch resolved Polymarket markets, build features, train + save model
  2. predict() — load saved model, return calibrated YES probability for a live market

Feature set (all available without extra API calls):
  - yes_price            : market implied prob at time of trade
  - liquidity_usdc       : order book depth
  - volume_24h_usdc      : 24h trading volume
  - days_to_resolution   : time pressure
  - sentiment_composite  : weighted sentiment score from research_agent
  - category_encoded     : integer-encoded market category
  - price_momentum       : how much yes_price moved in last 24h (if available)
  - volume_liquidity_ratio: volume/liquidity — proxy for market efficiency
  - is_near_50           : binary flag, market near 50% (most exploitable)

The model predicts: P(YES resolves = 1)
Edge = model_prob - market_yes_price
"""

import os
import json
import time
import logging
import pickle
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/xgboost_model.pkl")
ENCODER_PATH = Path("models/category_encoder.json")
GAMMA_URL = "https://gamma-api.polymarket.com"

# ─── CATEGORY ENCODER ────────────────────────────────────────────────────────

KNOWN_CATEGORIES = [
    "politics", "crypto", "sports", "economics",
    "science", "entertainment", "world", "other"
]

def encode_category(category: str) -> int:
    cat = (category or "other").lower().strip()
    for i, known in enumerate(KNOWN_CATEGORIES):
        if known in cat:
            return i
    return len(KNOWN_CATEGORIES) - 1  # "other"


# ─── FEATURE BUILDER ─────────────────────────────────────────────────────────

def build_features(market: dict, sentiment_composite: float = 0.0) -> np.ndarray:
    """
    Build feature vector for a single market.
    Must match exactly the features used during training.
    Called by predict() at runtime for live markets.
    """
    yes_price   = float(market.get("yes_price", 0.5))
    liquidity   = float(market.get("liquidity_usdc", 0))
    volume      = float(market.get("volume_24h_usdc", 0))
    days        = float(market.get("days_to_resolution") or 15)
    category    = encode_category(market.get("category", "other"))

    # Derived features
    vol_liq_ratio = volume / max(liquidity, 1.0)
    is_near_50    = 1 if 0.35 <= yes_price <= 0.65 else 0
    log_liquidity = np.log1p(liquidity)
    log_volume    = np.log1p(volume)

    return np.array([
        yes_price,
        log_liquidity,
        log_volume,
        days,
        sentiment_composite,
        float(category),
        vol_liq_ratio,
        is_near_50,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "yes_price",
    "log_liquidity",
    "log_volume",
    "days_to_resolution",
    "sentiment_composite",
    "category_encoded",
    "vol_liq_ratio",
    "is_near_50",
]


# ─── HISTORICAL DATA FETCHER ─────────────────────────────────────────────────

def _get(url: str, params: dict = None, retries: int = 3) -> list | dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"Request failed ({attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    return None


def fetch_resolved_markets(limit: int = 1000, min_volume: float = 500.0) -> list[dict]:
    """
    Fetch resolved (closed) markets from Gamma API.
    These have a known outcome — our training labels.

    Returns list of dicts with all fields needed to build features + label.
    """
    resolved = []
    offset   = 0
    page_size = 100

    logger.info(f"Fetching resolved markets (target: {limit})...")

    while len(resolved) < limit:
        params = {
            "closed":     "true",
            "limit":      page_size,
            "offset":     offset,
            "order":      "volume",
            "ascending":  "false",
        }
        data = _get(f"{GAMMA_URL}/markets", params=params)
        if not data or not isinstance(data, list):
            logger.warning("Empty response from Gamma API, stopping fetch")
            break

        for m in data:
            # Skip non-binary markets
            try:
                outcomes = json.loads(m.get("outcomes", "[]")) if isinstance(m.get("outcomes"), str) else (m.get("outcomes") or [])
            except (json.JSONDecodeError, TypeError):
                continue
            if len(outcomes) < 2:
                continue

            # Determine resolution from outcomePrices — price "1" = winner
            try:
                prices = json.loads(m.get("outcomePrices", "[]")) if isinstance(m.get("outcomePrices"), str) else (m.get("outcomePrices") or [])
            except (json.JSONDecodeError, TypeError):
                continue
            if len(prices) < 2:
                continue
            # A resolved market has one outcome at 1 and the other at 0
            try:
                p0, p1 = float(prices[0]), float(prices[1])
            except (ValueError, TypeError):
                continue
            if not (p0 > 0.9 or p1 > 0.9):
                continue  # not clearly resolved

            # Need minimum volume for signal quality
            vol = float(m.get("volume") or m.get("volumeNum") or 0)
            if vol < min_volume:
                continue

            resolved.append(m)

        logger.info(f"Fetched {len(resolved)} resolved markets so far (offset={offset})")
        offset += page_size

        if len(data) < page_size:
            break   # no more pages

        time.sleep(0.3)  # rate limit courtesy

    logger.info(f"Total resolved markets fetched: {len(resolved)}")
    return resolved


def parse_resolution_label(market: dict) -> Optional[int]:
    """
    Parse the market resolution into a binary label.
    Returns 1 if first outcome won (YES), 0 if second outcome won (NO), None if ambiguous.

    Gamma API encodes resolution via outcomePrices: the winning outcome gets
    price "1" and the loser gets "0". outcomes[0] is typically YES.
    """
    # Primary method: outcomePrices — the resolved price tells us the winner
    try:
        prices_raw = market.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else (prices_raw or [])
        if len(prices) >= 2:
            p0, p1 = float(prices[0]), float(prices[1])
            if p0 > 0.9 and p1 < 0.1:
                return 1   # first outcome (YES) won
            if p1 > 0.9 and p0 < 0.1:
                return 0   # second outcome (NO) won
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: explicit resolution field (some older markets)
    resolution = (
        market.get("resolution") or
        market.get("winner") or
        market.get("resolvedOutcome") or ""
    ).upper().strip()

    if resolution in ("YES", "1", "TRUE", "WIN"):
        return 1
    if resolution in ("NO", "0", "FALSE", "LOSE", "LOSS"):
        return 0

    return None  # can't determine — skip this market


def extract_yes_price_at_close(market: dict) -> float:
    """
    Extract the YES price near market close (the price the model would have seen).
    For resolved markets, outcomePrices is 1/0 (post-resolution), so we use
    lastTradePrice or bestBid as the pre-resolution price.
    """
    # lastTradePrice is the most recent trade before resolution
    ltp = market.get("lastTradePrice")
    if ltp is not None:
        try:
            price = float(ltp)
            if 0.01 <= price <= 0.99:
                return price
        except (ValueError, TypeError):
            pass

    # bestBid is another proxy
    for field in ["bestBid", "bestAsk"]:
        val = market.get(field)
        if val is not None:
            try:
                price = float(val)
                if 0.01 <= price <= 0.99:
                    return price
            except (ValueError, TypeError):
                pass

    return 0.5  # unknown — center prior


def build_training_row(market: dict) -> Optional[dict]:
    """Convert a raw Gamma resolved market into a training row."""
    label = parse_resolution_label(market)
    if label is None:
        return None

    yes_price = extract_yes_price_at_close(market)

    # Skip resolved-at-extreme markets (no edge to learn from)
    if yes_price > 0.97 or yes_price < 0.03:
        return None

    # Parse end date for days_to_resolution at time of listing
    start_date = market.get("startDate") or market.get("createdAt") or ""
    end_date   = market.get("endDate") or market.get("endDateIso") or ""
    try:
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end   = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        days  = max(0, (end - start).days)
    except Exception:
        days = 14  # default

    vol       = float(market.get("volume") or market.get("volumeNum") or 0)
    liquidity = float(market.get("liquidityNum") or market.get("liquidity") or vol * 0.3)
    category  = ""
    tags      = market.get("tags") or []
    if isinstance(tags, list) and tags:
        category = tags[0].get("slug", "") if isinstance(tags[0], dict) else str(tags[0])

    return {
        "yes_price":          yes_price,
        "liquidity_usdc":     liquidity,
        "volume_24h_usdc":    vol / max(days, 1),  # approximate daily volume
        "days_to_resolution": days,
        "sentiment_composite": 0.0,  # no historical sentiment — set neutral
        "category":           category,
        "label":              label,
    }


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train(min_samples: int = 200, n_estimators: int = 300,
          save: bool = True) -> dict:
    """
    Full training pipeline:
      1. Fetch resolved markets from Gamma API
      2. Build feature matrix
      3. Train XGBoost with cross-validation
      4. Save model + report metrics

    Returns dict with training metrics.
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import (
            roc_auc_score, brier_score_loss, log_loss
        )
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Run: pip install xgboost scikit-learn")

    # ── 1. Fetch data ──
    raw_markets = fetch_resolved_markets(limit=2000, min_volume=200.0)
    if len(raw_markets) < min_samples:
        raise ValueError(
            f"Only {len(raw_markets)} resolved markets found. "
            f"Need at least {min_samples}. The model cannot be trained yet — "
            f"bot will use heuristic fallback until more data is available."
        )

    # ── 2. Build features ──
    rows = []
    for m in raw_markets:
        row = build_training_row(m)
        if row:
            rows.append(row)

    logger.info(f"Built {len(rows)} valid training rows from {len(raw_markets)} markets")

    if len(rows) < min_samples:
        raise ValueError(f"Only {len(rows)} usable rows after filtering. Need {min_samples}.")

    df = pd.DataFrame(rows)

    X = np.column_stack([
        df["yes_price"].values,
        np.log1p(df["liquidity_usdc"].values),
        np.log1p(df["volume_24h_usdc"].values),
        df["days_to_resolution"].values,
        df["sentiment_composite"].values,
        df["category"].map(encode_category).values.astype(float),
        (df["volume_24h_usdc"] / df["liquidity_usdc"].clip(lower=1)).values,
        ((df["yes_price"] >= 0.35) & (df["yes_price"] <= 0.65)).astype(float).values,
    ])
    y = df["label"].values.astype(int)

    pos_rate = y.mean()
    logger.info(f"Label balance: {pos_rate:.1%} YES resolutions")

    # ── 3. Train XGBoost ──
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    base_model = xgb.XGBClassifier(
        n_estimators      = n_estimators,
        max_depth         = 4,           # shallow — avoid overfit on small dataset
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        eval_metric       = "logloss",
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
    )

    # Platt scaling calibration — critical for probability reliability
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)

    # ── 4. Cross-validate ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores    = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    brier_scores  = cross_val_score(model, X, y, cv=cv, scoring="neg_brier_score")

    logger.info(f"CV ROC-AUC:    {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    logger.info(f"CV Brier score: {(-brier_scores.mean()):.3f} ± {brier_scores.std():.3f}")

    # ── 5. Fit on full data ──
    model.fit(X, y)

    metrics = {
        "n_samples":        len(rows),
        "pos_rate":         float(pos_rate),
        "cv_auc_mean":      float(auc_scores.mean()),
        "cv_auc_std":       float(auc_scores.std()),
        "cv_brier_mean":    float(-brier_scores.mean()),
        "trained_at":       datetime.now(timezone.utc).isoformat(),
        "n_estimators":     n_estimators,
        "feature_names":    FEATURE_NAMES,
    }

    # ── 6. Save ──
    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "metrics": metrics}, f)
        with open(ENCODER_PATH, "w") as f:
            json.dump(KNOWN_CATEGORIES, f)
        logger.info(f"Model saved → {MODEL_PATH}")
        logger.info(f"Metrics: AUC={metrics['cv_auc_mean']:.3f}, "
                    f"Brier={metrics['cv_brier_mean']:.3f}, "
                    f"n={metrics['n_samples']}")

    return metrics


# ─── INFERENCE ────────────────────────────────────────────────────────────────

_cached_model = None

def load_model():
    """Load model from disk, with in-process cache."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    _cached_model = data["model"]
    logger.info(f"XGBoost model loaded. Trained on {data['metrics'].get('n_samples')} samples, "
                f"AUC={data['metrics'].get('cv_auc_mean', 0):.3f}")
    return _cached_model


def predict(market: dict, sentiment_composite: float = 0.0) -> float:
    """
    Return calibrated P(YES) for a live market.
    Falls back to heuristic if model not trained yet.

    Args:
        market:              normalized market dict from DB
        sentiment_composite: weighted composite from research_agent (-1 to +1)

    Returns:
        float: probability of YES resolution (0.0 - 1.0)
    """
    model = load_model()

    if model is None:
        # Heuristic fallback: market price ± sentiment adjustment
        yes_price = float(market.get("yes_price", 0.5))
        from config import RISK
        adj = sentiment_composite * RISK.sentiment_weight
        prob = np.clip(yes_price + adj, 0.02, 0.98)
        logger.debug(f"Heuristic fallback: yes_price={yes_price:.3f} "
                     f"sentiment_adj={adj:+.3f} → prob={prob:.3f}")
        return float(prob)

    features = build_features(market, sentiment_composite).reshape(1, -1)
    prob = model.predict_proba(features)[0][1]   # P(class=1) = P(YES)
    return float(np.clip(prob, 0.02, 0.98))


def model_info() -> dict:
    """Return metadata about the currently loaded model."""
    if not MODEL_PATH.exists():
        return {"status": "not_trained", "path": str(MODEL_PATH)}
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return {"status": "loaded", **data.get("metrics", {})}


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "train":
        print("Training XGBoost model on resolved Polymarket data...")
        try:
            metrics = train()
            print(f"\n✅ Training complete")
            print(f"   Samples:    {metrics['n_samples']}")
            print(f"   ROC-AUC:    {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
            print(f"   Brier:      {metrics['cv_brier_mean']:.3f}")
            print(f"   Model path: {MODEL_PATH}")
            print(f"\n{'='*50}")
            print("AUC interpretation:")
            print("  > 0.65  — model has genuine predictive signal ✓")
            print("  0.55-0.65 — marginal, use with caution")
            print("  ~0.50  — no better than random, stay in heuristic mode")
        except ValueError as e:
            print(f"\n⚠️  {e}")
            print("Bot will use heuristic fallback (market price ± sentiment).")
            print("Re-run training after more markets resolve.")

    elif cmd == "info":
        info = model_info()
        print(json.dumps(info, indent=2))

    else:
        print("Usage: python models/xgboost_model.py [train|info]")
