"""
agents/risk_agent.py — Risk management and Kelly criterion sizing.

Checks:
  1. Daily loss limit not breached
  2. Total open exposure within bounds
  3. Kelly-sized bet, capped at config limits

Public API (called by run_bot._process_market):
    approve_trade(prediction, market) → (approved: bool, size_usdc: float)
"""

import logging

from config import RISK
from data.db import get_daily_pnl, get_open_exposure
from data.normalizer import kelly_bet

logger = logging.getLogger(__name__)


def approve_trade(prediction: dict, market: dict) -> tuple[bool, float]:
    """
    Evaluate whether a trade should proceed and how large it should be.

    Args:
        prediction: Dict from predict_agent (claude_prob, direction, edge_pct…).
        market:     Market dict from DB (yes_price, no_price…).

    Returns:
        (approved, size_usdc). approved=False → skip this trade.
    """
    condition_id = market["condition_id"]
    direction = prediction["direction"]
    claude_prob = prediction["claude_prob"]

    # ── 1. Daily loss limit ───────────────────────────────────────────────
    daily_pnl = get_daily_pnl()
    if daily_pnl <= -RISK.daily_loss_limit_usdc:
        logger.warning(
            f"[{condition_id[:8]}] REJECTED — daily loss limit "
            f"(pnl=${daily_pnl:.2f}, limit=-${RISK.daily_loss_limit_usdc:.2f})"
        )
        return False, 0.0

    # ── 2. Kelly sizing ───────────────────────────────────────────────────
    if direction == "YES":
        prob = claude_prob
        price = market.get("yes_price", 0.5)
    else:
        prob = 1.0 - claude_prob
        price = market.get("no_price", 0.5)

    size = kelly_bet(
        prob=prob,
        market_price=price,
        bankroll=RISK.bankroll_usdc,
        fraction=RISK.kelly_fraction,
        max_bet_pct=RISK.max_kelly_bet_pct,
    )

    if size <= 0:
        logger.info(f"[{condition_id[:8]}] REJECTED — Kelly size ≤ 0 (no edge)")
        return False, 0.0

    # Cap at absolute max single position
    size = min(size, RISK.max_single_position_usdc)

    # ── 3. Exposure check ─────────────────────────────────────────────────
    current_exposure = get_open_exposure()
    if current_exposure + size > RISK.max_total_exposure_usdc:
        available = RISK.max_total_exposure_usdc - current_exposure
        if available >= 1.0:
            size = min(size, available)
            logger.info(
                f"[{condition_id[:8]}] Reduced size to ${size:.2f} "
                f"to fit exposure limit"
            )
        else:
            logger.info(f"[{condition_id[:8]}] REJECTED — no exposure room")
            return False, 0.0

    logger.info(
        f"[{condition_id[:8]}] APPROVED: {direction} ${size:.2f} "
        f"(prob={prob:.3f} price={price:.3f} "
        f"daily_pnl=${daily_pnl:.2f} exposure=${current_exposure:.2f})"
    )

    return True, round(size, 2)
