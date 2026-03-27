"""
agents/execute_agent.py — Order placement via py-clob-client.

Authenticates with the Polymarket CLOB API, places limit orders,
persists trade records, and polls for fill status.

Public API (called by run_bot._process_market):
    place_order(market, prediction, size_usdc) → order_id | None
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs

from config import (
    POLYMARKET_CLOB_URL,
    POLYMARKET_GAMMA_URL,
    POLYMARKET_CHAIN_ID,
    POLYMARKET_API_KEY,
    POLYMARKET_API_SECRET,
    POLYMARKET_API_PASSPHRASE,
    POLYMARKET_PRIVATE_KEY,
    RISK,
)
from data.db import save_trade, update_trade

logger = logging.getLogger(__name__)

_clob_client: Optional[ClobClient] = None


def _get_client() -> Optional[ClobClient]:
    """Lazy-initialize and authenticate the CLOB client."""
    global _clob_client
    if _clob_client is not None:
        return _clob_client

    if not POLYMARKET_PRIVATE_KEY:
        logger.error("No POLYMARKET_PRIVATE_KEY — cannot place orders")
        return None

    try:
        _clob_client = ClobClient(
            POLYMARKET_CLOB_URL,
            key=POLYMARKET_PRIVATE_KEY,
            chain_id=POLYMARKET_CHAIN_ID,
        )

        if POLYMARKET_API_KEY and POLYMARKET_API_SECRET and POLYMARKET_API_PASSPHRASE:
            _clob_client.set_api_creds(ApiCreds(
                api_key=POLYMARKET_API_KEY,
                api_secret=POLYMARKET_API_SECRET,
                api_passphrase=POLYMARKET_API_PASSPHRASE,
            ))
        else:
            _clob_client.set_api_creds(_clob_client.derive_api_key())

        logger.info("CLOB client authenticated")
        return _clob_client

    except Exception as e:
        logger.error(f"CLOB client init failed: {e}")
        _clob_client = None  # reset so next call retries cleanly
        return None


def _poll_fill(order_id: str, trade_id: int,
               max_polls: int = 10, interval: float = 3.0) -> str:
    """
    Poll for order fill status and update the trade record.

    Returns final status string: 'filled', 'cancelled', 'expired', or 'pending'.
    """
    client = _get_client()
    if client is None:
        return "error"

    for attempt in range(max_polls):
        try:
            order = client.get_order(order_id)
            status = order.get("status", "unknown").upper()

            if status in ("MATCHED", "FILLED"):
                fill_price = order.get("associate_trades", [{}])
                avg_price = order.get("price")
                update_trade(trade_id, status="filled",
                             avg_fill_price=avg_price)
                logger.info(f"Order {order_id} filled @ {avg_price}")
                return "filled"

            if status in ("CANCELLED", "EXPIRED"):
                update_trade(trade_id, status=status.lower())
                logger.info(f"Order {order_id} {status.lower()}")
                return status.lower()

            logger.debug(f"Order {order_id}: {status} (poll {attempt + 1})")

        except Exception as e:
            logger.warning(f"Poll error for {order_id}: {e}")

        time.sleep(interval)

    logger.warning(f"Order {order_id} still pending after {max_polls} polls")
    return "pending"


def reconcile_pending_orders() -> int:
    """
    Re-check all trades with status='pending' against the CLOB API and
    update the DB if they have since been filled or cancelled.

    Called at the top of every trading cycle.  Returns the number of
    trades whose status was updated.
    """
    from data.db import get_open_trades

    client = _get_client()
    if client is None:
        return 0

    pending = [t for t in get_open_trades() if t["status"] == "pending"]
    if not pending:
        return 0

    updated = 0
    for trade in pending:
        order_id = trade.get("order_id")
        trade_id = trade.get("id")
        if not order_id or not trade_id:
            continue
        try:
            order = client.get_order(order_id)
        except AttributeError:
            # py-clob-client raises AttributeError when the CLOB returns null/empty
            # data for archived or already-settled orders.  Treat as filled.
            update_trade(trade_id, status="filled")
            logger.info(
                f"[reconcile] Trade #{trade_id}: CLOB returned null "
                f"(archived/settled) → marked filled"
            )
            updated += 1
            continue
        except Exception as e:
            logger.warning(f"[reconcile] Error fetching order for trade #{trade_id}: {e}")
            continue

        try:
            if order is None:
                # Explicit None return — same treatment as above
                update_trade(trade_id, status="filled")
                logger.info(
                    f"[reconcile] Trade #{trade_id} order not found on CLOB "
                    f"→ marked filled"
                )
                updated += 1
                continue
            status = order.get("status", "unknown").upper()

            if status in ("MATCHED", "FILLED"):
                avg_price = order.get("price")
                update_trade(trade_id, status="filled", avg_fill_price=avg_price)
                logger.info(
                    f"[reconcile] Trade #{trade_id} order {order_id[:12]}… "
                    f"→ filled @ {avg_price}"
                )
                updated += 1
            elif status in ("CANCELLED", "EXPIRED"):
                update_trade(trade_id, status=status.lower())
                logger.info(
                    f"[reconcile] Trade #{trade_id} order {order_id[:12]}… "
                    f"→ {status.lower()}"
                )
                updated += 1
            else:
                logger.debug(
                    f"[reconcile] Trade #{trade_id} still {status}"
                )
        except Exception as e:
            logger.warning(f"[reconcile] Error processing trade #{trade_id}: {e}")

    if updated:
        logger.info(f"[reconcile] Updated {updated}/{len(pending)} pending trades")
    return updated


def settle_resolved_positions() -> int:
    """
    For every filled trade with no P&L yet, query the Gamma API to see if
    the market has resolved.  When it has, calculate P&L and write
    pnl_usdc + settled_at so that:
      - The position is removed from open exposure
      - The learn agent can post-mortem it

    P&L logic (both YES and NO directions are symmetric):
      correct:  payout = size / fill_price  →  pnl = size * (1/fill_price − 1)
      wrong:    payout = 0                  →  pnl = −size
    """
    from data.db import get_conn, get_latest_prediction

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM trades
            WHERE status = 'filled'
              AND pnl_usdc IS NULL
              AND settled_at IS NULL
        """).fetchall()
        trades = [dict(r) for r in rows]

    if not trades:
        return 0

    settled = 0
    for trade in trades:
        condition_id = trade["condition_id"]
        trade_id     = trade["id"]
        direction    = (trade.get("direction") or "YES").upper()
        size_usdc    = trade.get("size_usdc") or 0.0

        try:
            resp = requests.get(
                f"{POLYMARKET_GAMMA_URL}/markets",
                params={"condition_ids": condition_id},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                continue

            mkt = data[0] if isinstance(data, list) else data

            # Market must be closed and effectively resolved.
            # Accept UMA "resolved" or clear 0/1 prices on a closed market
            # (prices can reach definitiveness before UMA formally resolves).
            raw_prices_raw = mkt.get("outcomePrices", "[]")
            raw_outcomes_raw = mkt.get("outcomes", '["Yes","No"]')

            import json as _json
            if isinstance(raw_prices_raw, str):
                raw_prices = _json.loads(raw_prices_raw)
            else:
                raw_prices = raw_prices_raw
            if isinstance(raw_outcomes_raw, str):
                raw_outcomes = _json.loads(raw_outcomes_raw)
            else:
                raw_outcomes = raw_outcomes_raw

            uma_status = mkt.get("umaResolutionStatus") or ""
            clear_winner = any(str(p).strip() in ("1", "1.0") for p in raw_prices)
            is_closed = mkt.get("closed", False)

            if not (uma_status == "resolved" or (is_closed and clear_winner)):
                continue  # market still genuinely open

            # Derive YES/NO resolution from the outcome with price "1"
            resolution = None
            for outcome, price in zip(raw_outcomes, raw_prices):
                if str(price).strip() in ("1", "1.0"):
                    resolution = outcome.strip().upper()  # "YES" or "NO"
                    break

            if not resolution:
                continue

            # Determine fill price — use actual avg fill, else prediction entry price
            fill_price = trade.get("avg_fill_price")
            if not fill_price:
                pred = get_latest_prediction(condition_id)
                if pred:
                    yp = pred.get("market_yes_price", 0.5)
                    fill_price = yp if direction == "YES" else round(1.0 - yp, 6)

            # Calculate P&L
            correct = (resolution == direction)
            if correct and fill_price and fill_price > 0:
                pnl = round(size_usdc * (1.0 / fill_price - 1.0), 4)
            else:
                pnl = round(-size_usdc, 4)

            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            update_trade(trade_id, pnl_usdc=pnl, settled_at=now_str)
            logger.info(
                f"[settle] Trade #{trade_id} {direction} on {condition_id[:8]}… "
                f"resolved {resolution} → P&L ${pnl:+.2f}"
            )
            settled += 1

        except Exception as e:
            logger.warning(f"[settle] Error checking trade #{trade_id}: {e}")

    if settled:
        logger.info(f"[settle] {settled} position(s) settled with P&L")
    return settled


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def place_order(market: dict, prediction: dict,
                size_usdc: float) -> Optional[str]:
    """
    Place a limit order on Polymarket and persist to trades table.

    Args:
        market:     Market dict (yes_token_id, no_token_id, prices).
        prediction: Prediction dict (direction, prediction_id, claude_prob).
        size_usdc:  Position size in USDC (from risk_agent).

    Returns:
        order_id string on success, None on failure.
    """
    condition_id = market["condition_id"]
    direction = prediction["direction"]

    client = _get_client()
    if client is None:
        logger.error(f"[{condition_id[:8]}] No CLOB client — cannot execute")
        return None

    # Token + price
    if direction == "YES":
        token_id = market.get("yes_token_id")
        price = market.get("yes_price", 0.5)
    else:
        token_id = market.get("no_token_id")
        price = market.get("no_price", 0.5)

    if not token_id:
        logger.error(f"[{condition_id[:8]}] No token_id for {direction}")
        return None
    if price <= 0:
        logger.error(f"[{condition_id[:8]}] Invalid price: {price}")
        return None

    quantity = size_usdc / price

    logger.info(
        f"[{condition_id[:8]}] Placing {direction} limit: "
        f"${size_usdc:.2f} @ {price:.4f} ({quantity:.2f} shares)"
    )

    try:
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=quantity,
            side="BUY",
        )
        resp = client.create_and_post_order(order_args)
        order_id = resp.get("orderID") or resp.get("order_id")

        if not order_id:
            logger.error(f"[{condition_id[:8]}] No order_id in response: {resp}")
            return None

        # Persist trade
        trade_row = {
            "condition_id":       condition_id,
            "prediction_id":      prediction.get("prediction_id"),
            "direction":          direction,
            "size_usdc":          size_usdc,
            "kelly_fraction_used": RISK.kelly_fraction,
            "bankroll_at_trade":  prediction.get("_bankroll", RISK.bankroll_usdc),
        }
        trade_id = save_trade(trade_row)
        update_trade(trade_id, order_id=order_id)

        logger.info(f"[{condition_id[:8]}] Order {order_id} placed (trade #{trade_id})")

        # Poll for fill
        _poll_fill(order_id, trade_id)

        return order_id

    except Exception as e:
        logger.error(f"[{condition_id[:8]}] Order placement failed: {e}")
        return None
