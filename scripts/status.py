#!/usr/bin/env python3
"""
scripts/status.py — Standalone monitoring dashboard for the prediction bot.
Run:  python3 scripts/status.py
"""

import os
import sys
import sqlite3
from datetime import datetime, date, timezone

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DB_PATH, RISK
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

DB = os.path.join(os.path.dirname(__file__), "..", DB_PATH)


def _conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch(query, params=()):
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


# ── helpers ───────────────────────────────────────────────────────────────────

def _days_until(iso_date):
    """Compute days from now until iso_date. Negative = already expired."""
    if not iso_date:
        return None
    try:
        end = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (end - now).days
    except Exception:
        return None


def _trunc(text, width=38):
    return text if len(text) <= width else text[: width - 1] + "…"


def _pnl_color(val):
    if val > 0:
        return "green"
    elif val < 0:
        return "red"
    return "dim"


# ── data queries ──────────────────────────────────────────────────────────────

def get_open_trades():
    """Trades that are pending or filled but not settled."""
    return _fetch(
        """
        SELECT t.id, t.condition_id, t.direction, t.size_usdc,
               t.avg_fill_price, t.status, t.placed_at, t.pnl_usdc,
               m.question, m.yes_price, m.no_price, m.end_date_iso, m.days_to_resolution
        FROM trades t
        LEFT JOIN markets m ON m.condition_id = t.condition_id
        WHERE t.status IN ('pending', 'filled')
          AND t.settled_at IS NULL
        ORDER BY t.placed_at DESC
        """
    )


def get_trade_summary():
    """Overall counts and totals."""
    rows = _fetch(
        """
        SELECT status,
               COUNT(*) AS cnt,
               COALESCE(SUM(size_usdc), 0) AS total_usdc
        FROM trades
        GROUP BY status
        """
    )
    return {r["status"]: r for r in rows}


def get_daily_pnl():
    """Sum of settled P&L for trades settled today."""
    today = date.today().isoformat()
    row = _fetch(
        "SELECT COALESCE(SUM(pnl_usdc), 0) AS pnl FROM trades WHERE DATE(settled_at) = ?",
        (today,),
    )
    return row[0]["pnl"] if row else 0.0


def get_recent_predictions(limit=10):
    return _fetch(
        """
        SELECT p.condition_id, p.claude_prob, p.market_yes_price,
               p.edge_pct, p.confidence, p.predicted_at,
               m.question
        FROM predictions p
        LEFT JOIN markets m ON m.condition_id = p.condition_id
        ORDER BY p.predicted_at DESC
        LIMIT ?
        """,
        (limit,),
    )


# ── display ───────────────────────────────────────────────────────────────────

def render():
    open_trades = get_open_trades()
    summary = get_trade_summary()
    daily_pnl = get_daily_pnl()
    predictions = get_recent_predictions()

    total_trades = sum(s["cnt"] for s in summary.values())
    open_exposure = sum(t["size_usdc"] for t in open_trades)
    bankroll = RISK.bankroll_usdc
    exposure_pct = (open_exposure / bankroll * 100) if bankroll else 0

    # ── header panel ──────────────────────────────────────────────────────
    header = Text()
    header.append(f"  Bankroll:        ${bankroll:,.2f}\n")
    header.append(f"  Open exposure:   ${open_exposure:,.2f} ({exposure_pct:.1f}%)\n")

    pnl_str = f"+${daily_pnl:.2f}" if daily_pnl >= 0 else f"-${abs(daily_pnl):.2f}"
    header.append("  Daily P&L:       ")
    header.append(pnl_str, style=_pnl_color(daily_pnl))
    header.append("\n")
    header.append(f"  Total trades:    {total_trades}")

    console.print()
    console.print(
        Panel(header, title="[bold]PREDICTION BOT STATUS[/bold]", width=56, padding=(1, 0))
    )

    # ── open positions table ──────────────────────────────────────────────
    pos_table = Table(
        title="OPEN POSITIONS",
        show_header=True,
        header_style="bold",
        width=56,
        show_edge=False,
        pad_edge=False,
    )
    pos_table.add_column("DIR", width=3)
    pos_table.add_column("Market", ratio=1)
    pos_table.add_column("Ends", justify="right", width=7)
    pos_table.add_column("Size", justify="right", width=7)
    pos_table.add_column("P&L", justify="right", width=9)

    for t in open_trades:
        question = _trunc(t["question"] or t["condition_id"][:10], 28)
        direction = t["direction"]
        size_str = f"${t['size_usdc']:.2f}"

        # Estimate unrealised P&L from current market price
        if t["status"] == "pending":
            pnl_display = Text("pending", style="dim")
        elif t["avg_fill_price"] and (t["yes_price"] is not None or t["no_price"] is not None):
            current = t["no_price"] if direction == "NO" else t["yes_price"]
            if current and t["avg_fill_price"]:
                change_pct = (current - t["avg_fill_price"]) / t["avg_fill_price"] * 100
                sign = "+" if change_pct >= 0 else ""
                pnl_display = Text(
                    f"{sign}{change_pct:.1f}%", style=_pnl_color(change_pct)
                )
            else:
                pnl_display = Text("—", style="dim")
        else:
            pnl_display = Text("—", style="dim")

        # End date display
        end_iso = t.get("end_date_iso")
        days = _days_until(end_iso)  # always computed fresh
        if end_iso:
            # Format as dd/mm
            end_str = datetime.fromisoformat(end_iso.replace("Z", "+00:00")).strftime("%d/%m")
            if days is None or days < 0:
                ends_display = Text(end_str, style="red")
            elif days == 0:
                ends_display = Text("today", style="red bold")
            elif days <= 3:
                ends_display = Text(end_str, style="yellow")
            else:
                ends_display = Text(end_str, style="dim")
        else:
            ends_display = Text("unknown", style="red")

        pos_table.add_row(direction, question, ends_display, size_str, pnl_display)

    if not open_trades:
        pos_table.add_row("", Text("No open positions", style="dim"), "", "", "")

    console.print(pos_table)
    console.print()

    # ── recent predictions ────────────────────────────────────────────────
    pred_table = Table(
        title="RECENT PREDICTIONS (last 10)",
        show_header=True,
        header_style="bold",
        width=56,
        show_edge=False,
        pad_edge=False,
    )
    pred_table.add_column("Market", ratio=1)
    pred_table.add_column("Edge", justify="right", width=7)
    pred_table.add_column("Conf", justify="right", width=7)
    pred_table.add_column("Time", justify="right", width=6)

    for p in predictions:
        question = _trunc(p["question"] or p["condition_id"][:10], 24)
        edge = p["edge_pct"] or 0
        edge_str = f"+{edge * 100:.1f}%" if edge >= 0 else f"{edge * 100:.1f}%"
        conf = (p["confidence"] or "—")[:3]
        ts = (p["predicted_at"] or "")
        time_str = ts[11:16] if len(ts) >= 16 else ts

        pred_table.add_row(
            question,
            Text(edge_str, style=_pnl_color(edge)),
            conf,
            time_str,
        )

    if not predictions:
        pred_table.add_row(Text("No predictions yet", style="dim"), "", "", "")

    console.print(pred_table)

    # ── service status ────────────────────────────────────────────────────
    import subprocess

    res = subprocess.run(
        ["systemctl", "is-active", "prediction-bot"],
        capture_output=True,
        text=True,
    )
    svc = res.stdout.strip()
    svc_style = "bold green" if svc == "active" else "bold red"
    console.print()
    console.print(f"  Service: [{svc_style}]{svc}[/{svc_style}]")
    console.print()


if __name__ == "__main__":
    render()
