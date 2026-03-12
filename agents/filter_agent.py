"""
agents/filter_agent.py — Market opportunity scoring and ranking.

Pulls active markets from the DB, scores each on volume, liquidity,
days_to_resolution, and sentiment variance, then returns the top N
candidates sorted by opportunity score.

Opportunity formula:
    (volume_score * 0.4) + (liquidity_score * 0.3) + (time_score * 0.3)

Public API:
    rank_markets(top_n) → list[dict]  (markets enriched w/ opportunity_score)
"""

import logging

from config import FILTER, RISK
from data.db import get_active_markets, get_latest_sentiment
from data.normalizer import normalize_score

logger = logging.getLogger(__name__)

# ─── NORMALIZATION BOUNDS ────────────────────────────────────────────────────

_VOL_MIN   = 500.0
_VOL_MAX   = 100_000.0
_LIQ_MIN   = 1_000.0
_LIQ_MAX   = 200_000.0
_DAYS_MIN  = 1
_DAYS_MAX  = 30


def _volume_score(market: dict) -> float:
    """Score 0-1 based on 24h volume. Higher = better."""
    return normalize_score(market.get("volume_24h_usdc", 0), _VOL_MIN, _VOL_MAX)


def _liquidity_score(market: dict) -> float:
    """Score 0-1 based on liquidity depth. Higher = better."""
    return normalize_score(market.get("liquidity_usdc", 0), _LIQ_MIN, _LIQ_MAX)


def _time_score(market: dict) -> float:
    """
    Score 0-1 based on days to resolution.
    Sweet spot is 3-14 days; very soon or very far out score lower.
    """
    days = market.get("days_to_resolution")
    if days is None:
        return 0.3
    if days <= 0:
        return 0.0
    if days <= 2:
        return 0.6
    if days <= 14:
        return 1.0
    if days <= 30:
        return 1.0 - 0.6 * ((days - 14) / 16)
    return 0.2


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
    """Composite opportunity score for a single market."""
    vol  = _volume_score(market)
    liq  = _liquidity_score(market)
    tim  = _time_score(market)
    bonus = _sentiment_bonus(market["condition_id"])

    score = (vol * 0.4) + (liq * 0.3) + (tim * 0.3) + bonus
    return round(score, 4)


def _passes_prefilter(market: dict) -> bool:
    """Quick reject before scoring."""
    if market.get("active") != 1 or market.get("closed") == 1:
        return False
    if market.get("liquidity_usdc", 0) < RISK.min_liquidity_usdc:
        return False
    yes_p = market.get("yes_price", 0.5)
    if yes_p > 0.95 or yes_p < 0.05:
        return False
    days = market.get("days_to_resolution")
    if days is not None and days > RISK.max_days_to_resolution:
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

    eligible.sort(key=lambda m: m["opportunity_score"], reverse=True)
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
