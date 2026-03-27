"""
agents/risk_agent.py — Risk management and Kelly criterion sizing.

Checks:
  1. Daily loss limit not breached
  2. Total open exposure within bounds
  3. Kelly-sized bet, capped at config limits
  4. Correlation check against open positions (Claude, only when 3+ open)

Public API (called by run_bot._process_market):
    approve_trade(prediction, market) → (approved: bool, size_usdc: float)
"""

import json
import logging
import os
import time

import anthropic
from web3 import Web3

from config import RISK, ANTHROPIC_API_KEY, CLAUDE_MODEL
from data.db import get_daily_pnl, get_open_exposure, get_conn
from data.normalizer import kelly_bet

logger = logging.getLogger(__name__)

# ─── LIVE BANKROLL ───────────────────────────────────────────────────────────

_USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
_WALLET = os.getenv("POLYMARKET_WALLET_ADDRESS", "")
_POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"
_ERC20_ABI = [{"inputs": [{"name": "account", "type": "address"}],
               "name": "balanceOf",
               "outputs": [{"name": "", "type": "uint256"}],
               "stateMutability": "view", "type": "function"}]

_cached_bankroll: float | None = None
_cached_at: float = 0.0
_CACHE_TTL = 270  # 4.5 min — refreshes roughly once per cycle


def get_live_bankroll() -> float:
    """
    Fetch on-chain USDC.e balance + open position value.
    Cached for ~5 minutes to avoid spamming the RPC.
    Falls back to config value on error.
    """
    global _cached_bankroll, _cached_at

    now = time.time()
    if _cached_bankroll is not None and (now - _cached_at) < _CACHE_TTL:
        return _cached_bankroll

    wallet_balance = _fetch_wallet_balance()
    open_value = get_open_exposure()  # USDC locked in open positions
    total = wallet_balance + open_value

    _cached_bankroll = total
    _cached_at = now
    logger.info(f"Bankroll: wallet=${wallet_balance:.2f} + open=${open_value:.2f} = ${total:.2f}")
    return total


def _fetch_wallet_balance() -> float:
    """Query on-chain USDC.e balance. Returns config fallback on error."""
    if not _WALLET:
        return RISK.bankroll_usdc
    try:
        w3 = Web3(Web3.HTTPProvider(_POLYGON_RPC, request_kwargs={"timeout": 10}))
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(_USDC_E), abi=_ERC20_ABI
        )
        raw = contract.functions.balanceOf(
            Web3.to_checksum_address(_WALLET)
        ).call()
        return raw / 1e6
    except Exception as e:
        logger.warning(f"Bankroll fetch failed, using config: {e}")
        return RISK.bankroll_usdc


def _get_long_exposure() -> float:
    """Total USDC in open positions with days_to_resolution > long_threshold_days."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(t.size_usdc), 0) AS total
            FROM trades t
            LEFT JOIN markets m ON m.condition_id = t.condition_id
            WHERE t.status IN ('pending', 'filled')
              AND t.settled_at IS NULL
              AND (m.days_to_resolution IS NULL OR m.days_to_resolution > ?)
            """,
            (RISK.long_threshold_days,),
        ).fetchone()
        return row["total"] if row else 0.0


def _check_correlation(new_market: dict, new_direction: str) -> tuple[bool, str]:
    """
    Ask Claude if the new trade conflicts with any open positions.
    Only fires when there are 3+ open positions to justify the API call.
    Returns (is_conflicted, reasoning). Defaults to (False, "") on any error.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT t.condition_id, t.direction, m.question
            FROM trades t
            LEFT JOIN markets m ON m.condition_id = t.condition_id
            WHERE t.status IN ('pending', 'filled')
              AND t.settled_at IS NULL
              AND t.condition_id != ?
            """,
            (new_market["condition_id"],),
        ).fetchall()

    open_positions = [dict(r) for r in rows]

    if len(open_positions) < 3:
        return False, ""

    open_list = "\n".join(
        f"- [{r['condition_id'][:8]}] {r['direction']} on: {r['question']}"
        for r in open_positions
    )

    prompt = (
        f'You are a prediction market risk manager.\n\n'
        f'New trade: {new_direction} on "{new_market["question"]}"\n\n'
        f'Open positions:\n{open_list}\n\n'
        f'A conflict means the new trade and an existing position cannot both profit — '
        f'they bet on contradictory outcomes of the same underlying event.\n'
        f'Conflicts: "ceasefire YES" + "conflict continues YES"; same market opposite sides.\n'
        f'NOT conflicts: "Oil $100 NO" + "Oil $150 NO" (independent price levels); '
        f'"Bitcoin $75k NO" + "Bitcoin $80k NO" (both can simultaneously be true).\n\n'
        f'Return ONLY valid JSON (no markdown):\n'
        f'{{"conflicts": ["8-char-id", ...], "reasoning": "brief"}}\n'
        f'No conflicts: {{"conflicts": [], "reasoning": "no conflicts"}}'
    )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        result = json.loads(response.content[0].text.strip())
        conflicts = result.get("conflicts", [])
        reasoning = result.get("reasoning", "")
        if conflicts:
            return True, reasoning
    except Exception as e:
        logger.warning(f"Correlation check failed, allowing trade: {e}")

    return False, ""


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

    bankroll = get_live_bankroll()

    raw_kelly_f = max(0, ((1 / price - 1) * prob - (1 - prob)) / (1 / price - 1)) if price < 1 else 0
    logger.info(
        f"[{condition_id[:8]}] Kelly inputs: prob={prob:.4f} price={price:.4f} "
        f"bankroll=${bankroll:.2f} fraction={RISK.kelly_fraction} "
        f"max_bet_pct={RISK.max_kelly_bet_pct} | "
        f"raw_kelly_f={raw_kelly_f:.4f} fractional_kelly_f={raw_kelly_f * RISK.kelly_fraction:.4f} "
        f"uncapped_bet=${bankroll * raw_kelly_f * RISK.kelly_fraction:.2f} "
        f"cap=${bankroll * RISK.max_kelly_bet_pct:.2f}"
    )

    size = kelly_bet(
        prob=prob,
        market_price=price,
        bankroll=bankroll,
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

    # ── 3b. Duration-aware exposure cap ───────────────────────────────────
    market_days = market.get("days_to_resolution")
    is_long = (market_days is None) or (market_days > RISK.long_threshold_days)
    if is_long:
        long_exposure = _get_long_exposure()
        max_long = RISK.max_total_exposure_usdc * RISK.max_long_exposure_pct
        if long_exposure + size > max_long:
            available = max_long - long_exposure
            if available >= 1.0:
                size = min(size, available)
                logger.info(
                    f"[{condition_id[:8]}] Reduced to ${size:.2f} "
                    f"(long exposure cap: ${long_exposure:.2f}/${max_long:.2f})"
                )
            else:
                logger.info(
                    f"[{condition_id[:8]}] REJECTED — long exposure cap "
                    f"(${long_exposure:.2f}/${max_long:.2f})"
                )
                return False, 0.0

    # ── 4. Correlation check ──────────────────────────────────────────────
    conflicted, reason = _check_correlation(market, direction)
    if conflicted:
        logger.info(
            f"[{condition_id[:8]}] REJECTED — correlated with open position: {reason}"
        )
        return False, 0.0

    logger.info(
        f"[{condition_id[:8]}] APPROVED: {direction} ${size:.2f} "
        f"(prob={prob:.3f} price={price:.3f} "
        f"daily_pnl=${daily_pnl:.2f} exposure=${current_exposure:.2f})"
    )

    return True, round(size, 2)