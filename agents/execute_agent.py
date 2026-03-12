"""
agents/execute_agent.py — Order placement via py-clob-client.

Authenticates with the Polymarket CLOB API, places limit orders,
persists trade records, and polls for fill status.

Public API (called by run_bot._process_market):
    place_order(market, prediction, size_usdc) → order_id | None
"""

import logging
import time
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs

from config import (
    POLYMARKET_CLOB_URL,
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
            _clob_client.set_api_creds(ClobClient.ApiCreds(
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
            "bankroll_at_trade":  RISK.bankroll_usdc,
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
