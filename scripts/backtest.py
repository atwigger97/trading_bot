"""
scripts/backtest.py — Historical backtest using resolved Polymarket markets

Simulates the full pipeline on resolved markets to validate model performance.

Usage:
    python scripts/backtest.py                  # backtest with defaults
    python scripts/backtest.py --min-edge 0.03  # custom min edge
    python scripts/backtest.py --no-model        # heuristic-only mode
"""

import sys
import os
import json
import logging
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from files.xgboost_model import (
    fetch_resolved_markets, parse_resolution_label,
    extract_yes_price_at_close, build_features, load_model,
    encode_category, FEATURE_NAMES,
)
from data.normalizer import kelly_bet, edge as calc_edge
from config import RISK

logger = logging.getLogger(__name__)


def build_backtest_market(raw: dict) -> dict | None:
    """Convert a raw Gamma resolved market into a normalized market dict for prediction."""
    label = parse_resolution_label(raw)
    if label is None:
        return None

    yes_price = extract_yes_price_at_close(raw)
    if yes_price > 0.97 or yes_price < 0.03:
        return None

    start_date = raw.get("startDate") or raw.get("createdAt") or ""
    end_date = raw.get("endDate") or raw.get("endDateIso") or ""
    try:
        from datetime import datetime
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        days = max(1, (end - start).days)
    except Exception:
        days = 14

    vol = float(raw.get("volume") or raw.get("volumeNum") or 0)
    liquidity = float(raw.get("liquidityNum") or raw.get("liquidity") or vol * 0.3)
    category = ""
    tags = raw.get("tags") or []
    if isinstance(tags, list) and tags:
        category = tags[0].get("slug", "") if isinstance(tags[0], dict) else str(tags[0])

    return {
        "condition_id": raw.get("conditionId", ""),
        "question": raw.get("question", ""),
        "yes_price": yes_price,
        "no_price": 1.0 - yes_price,
        "liquidity_usdc": liquidity,
        "volume_24h_usdc": vol / max(days, 1),
        "days_to_resolution": days,
        "category": category,
        "label": label,  # 1=YES won, 0=NO won
    }


def run_backtest(min_edge: float = None, use_model: bool = True,
                 kelly_fraction: float = None, max_bet_pct: float = None,
                 bankroll: float = None, max_markets: int = 2000) -> dict:
    """
    Run a backtest on resolved Polymarket markets.

    Returns dict with performance metrics.
    """
    min_edge = min_edge if min_edge is not None else RISK.min_edge_pct
    kelly_frac = kelly_fraction if kelly_fraction is not None else RISK.kelly_fraction
    max_bp = max_bet_pct if max_bet_pct is not None else RISK.max_kelly_bet_pct
    starting_bankroll = bankroll if bankroll is not None else RISK.bankroll_usdc

    print(f"Fetching resolved markets...")
    raw_markets = fetch_resolved_markets(limit=max_markets, min_volume=200.0)
    print(f"Fetched {len(raw_markets)} resolved markets")

    model = load_model() if use_model else None
    if use_model and model is None:
        print("⚠️  No model found — falling back to heuristic")

    # Build market dicts
    markets = []
    for raw in raw_markets:
        m = build_backtest_market(raw)
        if m:
            markets.append(m)

    print(f"Usable markets for backtest: {len(markets)}")
    if not markets:
        return {"error": "No usable markets"}

    # Simulate trades
    current_bankroll = starting_bankroll
    trades = []
    wins = losses = skipped = 0
    total_pnl = 0.0

    for m in markets:
        label = m["label"]
        yes_price = m["yes_price"]

        # Get model probability
        if model is not None:
            features = build_features(m, sentiment_composite=0.0).reshape(1, -1)
            prob = float(np.clip(model.predict_proba(features)[0][1], 0.02, 0.98))
        else:
            # Heuristic: market price (no sentiment in backtest)
            prob = yes_price

        # Determine direction and edge
        yes_edge = prob - yes_price
        no_edge = (1 - prob) - (1 - yes_price)  # same magnitude, opposite sign

        if yes_edge >= min_edge:
            direction = "YES"
            trade_prob = prob
            trade_price = yes_price
        elif no_edge >= min_edge:
            direction = "NO"
            trade_prob = 1 - prob
            trade_price = 1 - yes_price
        else:
            skipped += 1
            continue

        # Kelly sizing
        size = kelly_bet(
            prob=trade_prob,
            market_price=trade_price,
            bankroll=current_bankroll,
            fraction=kelly_frac,
            max_bet_pct=max_bp,
        )
        size = min(size, RISK.max_single_position_usdc)

        if size <= 0:
            skipped += 1
            continue

        # Determine outcome
        won = (direction == "YES" and label == 1) or (direction == "NO" and label == 0)

        if won:
            # Payout: size / price (shares * $1 each)
            pnl = size * (1 / trade_price - 1)
            wins += 1
        else:
            pnl = -size
            losses += 1

        current_bankroll += pnl
        total_pnl += pnl

        trades.append({
            "question": m["question"][:60],
            "direction": direction,
            "prob": round(trade_prob, 3),
            "price": round(trade_price, 3),
            "edge": round(trade_prob - trade_price, 3),
            "size": round(size, 2),
            "won": won,
            "pnl": round(pnl, 2),
            "bankroll_after": round(current_bankroll, 2),
        })

        # Stop if bankrupt
        if current_bankroll <= 0:
            print("💀 Bankrupt!")
            break

    # Compute metrics
    n_trades = wins + losses
    win_rate = wins / n_trades if n_trades > 0 else 0
    avg_win = np.mean([t["pnl"] for t in trades if t["won"]]) if wins > 0 else 0
    avg_loss = np.mean([t["pnl"] for t in trades if not t["won"]]) if losses > 0 else 0
    roi = total_pnl / starting_bankroll if starting_bankroll > 0 else 0
    max_drawdown = 0.0
    peak = starting_bankroll
    for t in trades:
        bk = t["bankroll_after"]
        peak = max(peak, bk)
        dd = (peak - bk) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, dd)

    results = {
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": round(current_bankroll, 2),
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi * 100, 1),
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate * 100, 1),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "max_drawdown_pct": round(max_drawdown * 100, 1),
        "skipped": skipped,
        "total_markets": len(markets),
        "min_edge": min_edge,
        "kelly_fraction": kelly_frac,
        "used_model": model is not None,
    }

    return results, trades


def print_results(results: dict, trades: list, verbose: bool = False):
    """Pretty-print backtest results."""
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Model:              {'XGBoost' if results['used_model'] else 'Heuristic (no model)'}")
    print(f"  Markets tested:     {results['total_markets']}")
    print(f"  Trades placed:      {results['n_trades']} (skipped {results['skipped']})")
    print(f"  Min edge:           {results['min_edge']:.1%}")
    print(f"  Kelly fraction:     {results['kelly_fraction']}")
    print()
    print(f"  Starting bankroll:  ${results['starting_bankroll']:.2f}")
    print(f"  Ending bankroll:    ${results['ending_bankroll']:.2f}")
    print(f"  Total P&L:          ${results['total_pnl']:+.2f}")
    print(f"  ROI:                {results['roi_pct']:+.1f}%")
    print()
    print(f"  Win rate:           {results['win_rate']:.1f}% ({results['wins']}W / {results['losses']}L)")
    print(f"  Avg win:            ${results['avg_win']:+.2f}")
    print(f"  Avg loss:           ${results['avg_loss']:+.2f}")
    print(f"  Max drawdown:       {results['max_drawdown_pct']:.1f}%")
    print("=" * 60)

    if verbose and trades:
        print("\n  TRADE LOG (last 20):")
        print(f"  {'Dir':>3} {'Prob':>5} {'Price':>5} {'Edge':>5} {'Size':>6} {'P&L':>7} {'Bank':>8} Question")
        print("  " + "-" * 80)
        for t in trades[-20:]:
            w = "✓" if t["won"] else "✗"
            print(f"  {t['direction']:>3} {t['prob']:5.3f} {t['price']:5.3f} "
                  f"{t['edge']:+5.3f} ${t['size']:5.2f} ${t['pnl']:+6.2f} "
                  f"${t['bankroll_after']:7.2f} {w} {t['question'][:40]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Backtest prediction bot on resolved markets")
    parser.add_argument("--min-edge", type=float, default=None, help="Minimum edge to trade (default: config)")
    parser.add_argument("--no-model", action="store_true", help="Use heuristic only (skip XGBoost)")
    parser.add_argument("--kelly", type=float, default=None, help="Kelly fraction (default: config)")
    parser.add_argument("--bankroll", type=float, default=None, help="Starting bankroll (default: config)")
    parser.add_argument("--max-markets", type=int, default=2000, help="Max markets to fetch")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show trade log")
    args = parser.parse_args()

    results, trades = run_backtest(
        min_edge=args.min_edge,
        use_model=not args.no_model,
        kelly_fraction=args.kelly,
        bankroll=args.bankroll,
        max_markets=args.max_markets,
    )

    print_results(results, trades, verbose=args.verbose)
