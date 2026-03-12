"""
data/normalizer.py — Unified market format & display utilities

Provides helpers used by all agents to work with normalized market dicts.
"""

from typing import Optional
from datetime import datetime, timezone


def market_summary(m: dict) -> str:
    """Human-readable one-liner for a market. Used in logs + Claude prompts."""
    yes_p = m.get("yes_price", 0)
    days  = m.get("days_to_resolution", "?")
    vol   = m.get("volume_24h_usdc", 0)
    return (
        f"[{m.get('condition_id','?')[:8]}] "
        f"{m.get('question','?')[:80]} | "
        f"YES={yes_p:.0%} | "
        f"vol=${vol:,.0f} | "
        f"resolves in {days}d"
    )


def edge(our_prob: float, market_yes_price: float, direction: str = "YES") -> float:
    """
    Calculate edge for a given direction.
    edge > 0 means we have an advantage.
    """
    if direction == "YES":
        return our_prob - market_yes_price
    else:
        return (1 - our_prob) - (1 - market_yes_price)


def implied_prob(price: float) -> float:
    """Polymarket prices ARE probabilities (no overround to devig)."""
    return max(0.01, min(0.99, price))


def kelly_bet(prob: float, market_price: float,
              bankroll: float, fraction: float = 0.25,
              max_bet_pct: float = 0.05) -> float:
    """
    Fractional Kelly Criterion bet sizing.

    Args:
        prob:         Our estimated probability of YES
        market_price: Current market price of YES (= decimal odds baseline)
        bankroll:     Total USDC available
        fraction:     Kelly fraction (0.25 = quarter Kelly)
        max_bet_pct:  Hard cap as fraction of bankroll

    Returns:
        Bet size in USDC (0 if no edge)
    """
    if prob <= market_price:
        return 0.0   # no edge

    # Polymarket pays out at $1 per share → odds = 1/price - 1 (net)
    # Kelly formula: f* = (b*p - q) / b where b = net odds, p = win prob, q = loss prob
    b = (1 / market_price) - 1    # net odds per dollar risked
    q = 1 - prob
    kelly_f = (b * prob - q) / b

    kelly_f = max(0, kelly_f)
    kelly_f *= fraction             # fractional Kelly

    bet = bankroll * kelly_f
    bet = min(bet, bankroll * max_bet_pct)   # hard cap

    return round(bet, 2)


def format_market_for_claude(m: dict, sentiment: dict = None) -> str:
    """
    Format a market dict as a structured prompt block for Claude.
    Used by predict_agent and learn_agent.
    """
    lines = [
        f"Market: {m.get('question')}",
        f"Category: {m.get('category', 'unknown')}",
        f"Current YES price (market implied prob): {m.get('yes_price', 0):.1%}",
        f"Current NO price: {m.get('no_price', 0):.1%}",
        f"24h Volume: ${m.get('volume_24h_usdc', 0):,.0f} USDC",
        f"Liquidity: ${m.get('liquidity_usdc', 0):,.0f} USDC",
        f"Days to resolution: {m.get('days_to_resolution', 'unknown')}",
    ]

    if sentiment:
        lines.append("")
        lines.append("Sentiment data:")
        for source, score in sentiment.items():
            direction = "bullish (YES)" if score > 0.1 else \
                        "bearish (NO)" if score < -0.1 else "neutral"
            lines.append(f"  {source}: {score:+.2f} ({direction})")

    return "\n".join(lines)
