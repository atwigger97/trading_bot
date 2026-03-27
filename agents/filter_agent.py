"""
agents/filter_agent.py — Market opportunity scoring and ranking.

Pulls active markets from the DB, scores each on volume, liquidity,
days_to_resolution, and category, then returns the top N
candidates sorted by opportunity score.

Focuses on short-duration, high-accuracy categories:
  sports (1-day), crypto price targets (1-3 day),
  economic data releases (1-2 day), political deadlines (2-3 day).

Public API:
    rank_markets(top_n) → list[dict]  (markets enriched w/ opportunity_score)
"""

import logging

from config import FILTER, RISK
from data.db import get_active_markets, get_latest_sentiment
from data.normalizer import normalize_score

logger = logging.getLogger(__name__)

# ─── NORMALIZATION BOUNDS ────────────────────────────────────────────────────

_VOL_MIN   = 1_000.0
_VOL_MAX   = 100_000.0
_LIQ_MIN   = 2_000.0
_LIQ_MAX   = 200_000.0
_DAYS_MIN  = 1
_DAYS_MAX  = 3

# Category multipliers — prioritise categories with higher accuracy
_CATEGORY_MULTIPLIERS = {
    'sports':       1.3,
    'crypto':       1.2,
    'economics':    1.2,
    'politics':     0.9,   # open-ended geopolitics deprioritised
    'finance':      1.0,
    'entertainment': 0.5,  # avoid
    'science':      0.7,
    'technology':   0.7,
}


def _volume_score(market: dict) -> float:
    """Score 0-1 based on 24h volume. Higher = better."""
    return normalize_score(market.get("volume_24h_usdc", 0), _VOL_MIN, _VOL_MAX)


def _liquidity_score(market: dict) -> float:
    """Score 0-1 based on liquidity depth. Higher = better."""
    return normalize_score(market.get("liquidity_usdc", 0), _LIQ_MIN, _LIQ_MAX)


def _time_score(market: dict) -> float:
    """
    Score 0-1 based on days to resolution.
    Heavily favours same-day and short-duration markets.
    """
    days = market.get("days_to_resolution")
    if days is None or days <= 0:
        return 0.0
    if days <= 1:
        return 1.0    # same day — top priority
    if days <= 3:
        return 0.85   # 1-3 days
    if days <= 7:
        return 0.65   # weekly
    if days <= 14:
        return 0.35   # biweekly
    return 0.15           # monthly — deprioritised


def _horizon_bucket(days) -> str:
    """Classify days_to_resolution into a time bucket."""
    if days is None or days <= 0:
        return 'long'
    if days <= 1:
        return 'same_day'
    if days <= 3:
        return 'short'
    if days <= 7:
        return 'medium'
    return 'long'


def _sentiment_bonus(condition_id: str) -> float:
    """
    Small bonus for markets with strong sentiment signal.
    Strong signal → more predictable → better opportunity.
    """
    rows = get_latest_sentiment(condition_id)
    if not rows:
        return 0.0
    # average absolute score across sources
    avg_abs = sum(abs(r.get("score", 0)) for r in rows) / len(rows)
    if avg_abs > 0.5:
        return 0.15
    if avg_abs > 0.3:
        return 0.08
    return 0.0


def opportunity_score(market: dict) -> float:
    """Composite opportunity score with category + time horizon multipliers."""
    vol  = _volume_score(market)
    liq  = _liquidity_score(market)
    tim  = _time_score(market)
    bonus = _sentiment_bonus(market["condition_id"])

    base = (vol * 0.30) + (liq * 0.25) + (tim * 0.45) + bonus

    # Apply time-horizon multiplier
    days = market.get("days_to_resolution")
    bucket = _horizon_bucket(days)
    time_weight = RISK.time_horizon_weights.get(bucket, 1.0)

    # Apply category multiplier
    cat = (market.get("category") or "").lower().strip()
    cat_weight = _CATEGORY_MULTIPLIERS.get(cat, 0.7)

    return round(base * time_weight * cat_weight, 4)


def _passes_prefilter(market: dict) -> bool:
    """Quick reject before scoring."""
    if market.get("active") != 1 or market.get("closed") == 1:
        return False
    if market.get("liquidity_usdc", 0) < RISK.min_liquidity_usdc:
        return False

    yes_p = market.get("yes_price", 0.5)
    no_p  = market.get("no_price", 0.5)

    # No-edge price zones
    if yes_p > 0.85 or yes_p < 0.15:
        return False

    # Spread check — wide spread = illiquid/broken market
    spread_sum = yes_p + no_p
    if spread_sum < 0.90 or spread_sum > 1.10:
        return False

    days = market.get("days_to_resolution")
    if days is None or days <= 0:
        return False  # expired, resolves today, or unknown end date
    if days > RISK.max_days_to_resolution:
        return False

    # Category gate — skip entertainment/science junk
    cat = (market.get("category") or "").lower().strip()
    if cat in ("entertainment",):
        return False

    return True


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def rank_markets(top_n: int = None) -> list[dict]:
    """
    Pull active markets, score, rank, return top N candidates.

    Returns list of market dicts enriched with 'opportunity_score',
    sorted descending.
    """
    top_n = top_n or FILTER.top_n_candidates

    markets = get_active_markets(
        min_liquidity=RISK.min_liquidity_usdc,
        min_volume=RISK.min_volume_24h_usdc,
        max_days=RISK.max_days_to_resolution,
    )
    logger.info(f"Filter: {len(markets)} active markets from DB")

    eligible = [m for m in markets if _passes_prefilter(m)]
    logger.info(f"Filter: {len(eligible)} pass prefilter")

    for m in eligible:
        m["opportunity_score"] = opportunity_score(m)
        m["_horizon_bucket"] = _horizon_bucket(m.get("days_to_resolution"))

    # Sort: short-duration first at similar scores (bucket priority, then score)
    bucket_order = {'same_day': 0, 'short': 1, 'medium': 2, 'long': 3}
    eligible.sort(
        key=lambda m: (-m["opportunity_score"], bucket_order.get(m["_horizon_bucket"], 3))
    )
    candidates = eligible[:top_n]

    for m in candidates:
        logger.info(
            f"  ▸ [{m['condition_id'][:8]}] "
            f"score={m['opportunity_score']:.3f} "
            f"vol=${m.get('volume_24h_usdc', 0):,.0f} "
            f"liq=${m.get('liquidity_usdc', 0):,.0f} "
            f"days={m.get('days_to_resolution', '?')}"
        )

    return candidates
