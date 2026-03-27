"""
agents/notify_agent.py — WhatsApp notifications via CallMeBot (free).

Setup (one-time, 2 minutes):
  1. Add +34 644 52 74 21 to your WhatsApp contacts
  2. Send it: "I allow callmebot to send me messages"
  3. It replies with your API key

Credentials in .env:
  WHATSAPP_PHONE=+447911123456   (your number, with country code)
  WHATSAPP_APIKEY=123456         (from CallMeBot reply)

Public API:
    send_notification(message: str) -> None
    notify_trade_placed(market, prediction, size_usdc) -> None
    notify_trade_filled(market, prediction, size_usdc) -> None
    notify_daily_loss_limit(pnl: float) -> None
    notify_daily_summary() -> None
"""

import logging
import os
from urllib.parse import quote

import requests

from data.db import get_daily_pnl, get_open_exposure, get_performance_stats

logger = logging.getLogger(__name__)

_PHONE  = os.getenv("WHATSAPP_PHONE", "")
_APIKEY = os.getenv("WHATSAPP_APIKEY", "")
_URL    = "https://api.callmebot.com/whatsapp.php"


def send_notification(message: str):
    """Send WhatsApp message via CallMeBot. Fails silently if not configured."""
    if not _PHONE or not _APIKEY:
        return
    # Strip HTML tags — CallMeBot sends plain text
    plain = message.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
    try:
        requests.get(_URL, params={
            "phone":  _PHONE,
            "text":   plain,
            "apikey": _APIKEY,
        }, timeout=10)
    except Exception:
        pass  # never crash the bot for a notification


def notify_trade_placed(market: dict, prediction: dict, size_usdc: float):
    """Notify when a trade is placed."""
    question = market.get("question", "?")[:80]
    direction = prediction.get("direction", "?")
    edge = prediction.get("edge_pct", 0)
    confidence = prediction.get("confidence", "?")
    send_notification(
        f"\U0001f7e2 <b>Trade placed</b>: {direction} ${size_usdc:.2f}\n"
        f"<i>{question}</i>\n"
        f"Edge: {edge:+.1%} | Conf: {confidence}"
    )


def notify_trade_filled(market: dict, prediction: dict, size_usdc: float):
    """Notify when a trade fills."""
    question = market.get("question", "?")[:80]
    direction = prediction.get("direction", "?")
    send_notification(
        f"\u2705 <b>Filled</b>: {direction} ${size_usdc:.2f}\n"
        f"<i>{question}</i>"
    )


def notify_daily_loss_limit(pnl: float):
    """Notify when daily loss limit is hit."""
    send_notification(
        f"\U0001f534 <b>Daily loss limit hit. Bot paused.</b>\n"
        f"P&L: ${pnl:+.2f}"
    )


def notify_daily_summary():
    """Send daily performance summary."""
    pnl = get_daily_pnl()
    exposure = get_open_exposure()
    stats = get_performance_stats()
    send_notification(
        f"\U0001f4ca <b>Daily Summary</b>\n"
        f"Trades today: {stats['total_trades']}\n"
        f"P&L today: ${pnl:+.2f}\n"
        f"Open exposure: ${exposure:.2f}\n"
        f"Win rate: {stats['win_rate']:.1%}"
    )
