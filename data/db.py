"""
data/db.py — SQLite persistence layer
All agents read/write through this module.
"""

import sqlite3
import json
import logging
from datetime import datetime
from contextlib import contextmanager
from config import DB_PATH
import os

logger = logging.getLogger(__name__)

os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)


@contextmanager
def get_conn():
    """Yield a SQLite connection with WAL mode and auto-commit/rollback."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # concurrent reads
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables. Safe to call multiple times (idempotent)."""
    with get_conn() as conn:
        conn.executescript("""

        -- ── MARKETS ──────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS markets (
            condition_id        TEXT PRIMARY KEY,
            question            TEXT NOT NULL,
            category            TEXT,
            end_date_iso        TEXT,
            days_to_resolution  INTEGER,
            active              INTEGER DEFAULT 1,
            closed              INTEGER DEFAULT 0,
            volume_24h_usdc     REAL DEFAULT 0,
            liquidity_usdc      REAL DEFAULT 0,
            -- Outcome tokens (binary markets have YES/NO)
            yes_token_id        TEXT,
            no_token_id         TEXT,
            yes_price           REAL,   -- current market implied prob for YES
            no_price            REAL,
            -- Raw JSON for full market object
            raw_json            TEXT,
            last_updated        TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(active, closed);
        CREATE INDEX IF NOT EXISTS idx_markets_category ON markets(category);

        -- ── SENTIMENT SCORES ──────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS sentiment (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id    TEXT NOT NULL,
            source          TEXT NOT NULL,   -- 'twitter' | 'reddit' | 'rss'
            score           REAL NOT NULL,   -- -1.0 to +1.0
            post_count      INTEGER DEFAULT 0,
            keywords        TEXT,            -- JSON array of matched keywords
            raw_summary     TEXT,            -- Claude's sentiment summary
            scraped_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (condition_id) REFERENCES markets(condition_id)
        );

        CREATE INDEX IF NOT EXISTS idx_sentiment_market ON sentiment(condition_id, scraped_at);

        -- ── PREDICTIONS ──────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS predictions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id        TEXT NOT NULL,
            xgboost_prob        REAL,   -- raw XGBoost YES probability
            sentiment_adj_prob  REAL,   -- after sentiment adjustment
            claude_prob         REAL,   -- Claude's calibrated final probability
            market_yes_price    REAL,   -- market price at prediction time
            edge_pct            REAL,   -- claude_prob - market_yes_price
            confidence          TEXT,   -- 'high' | 'medium' | 'low'
            claude_reasoning    TEXT,   -- Claude's written reasoning
            predicted_at        TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (condition_id) REFERENCES markets(condition_id)
        );

        -- ── TRADES ──────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS trades (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id        TEXT NOT NULL,
            prediction_id       INTEGER,
            direction           TEXT NOT NULL,  -- 'YES' | 'NO'
            size_usdc           REAL NOT NULL,
            avg_fill_price      REAL,
            order_id            TEXT,           -- Polymarket order ID
            status              TEXT DEFAULT 'pending',  -- pending|filled|cancelled|failed
            kelly_fraction_used REAL,
            bankroll_at_trade   REAL,
            placed_at           TEXT DEFAULT (datetime('now')),
            settled_at          TEXT,
            pnl_usdc            REAL,           -- filled after resolution
            FOREIGN KEY (condition_id) REFERENCES markets(condition_id),
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(condition_id);

        -- ── LEARNINGS (post-mortem) ──────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS learnings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id        INTEGER NOT NULL,
            condition_id    TEXT NOT NULL,
            outcome         TEXT NOT NULL,   -- 'win' | 'loss'
            pnl_usdc        REAL,
            our_prob        REAL,
            market_prob     REAL,
            resolution      TEXT,            -- 'YES' | 'NO'
            error_analysis  TEXT,            -- Claude's post-mortem text
            feature_flags   TEXT,            -- JSON: which features were misleading
            created_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (trade_id) REFERENCES trades(id)
        );

        -- ── BANKROLL SNAPSHOTS ───────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS bankroll (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            usdc_balance    REAL NOT NULL,
            open_positions  REAL NOT NULL DEFAULT 0,
            total_equity    REAL NOT NULL,
            daily_pnl       REAL DEFAULT 0,
            snapshot_at     TEXT DEFAULT (datetime('now'))
        );

        """)
    logger.info(f"Database initialized at {DB_PATH}")


# ─── MARKET HELPERS ──────────────────────────────────────────────────────────

def upsert_market(market: dict):
    """Insert or update a normalized market dict."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO markets (
                condition_id, question, category, end_date_iso,
                days_to_resolution, active, closed,
                volume_24h_usdc, liquidity_usdc,
                yes_token_id, no_token_id, yes_price, no_price, raw_json, last_updated
            ) VALUES (
                :condition_id, :question, :category, :end_date_iso,
                :days_to_resolution, :active, :closed,
                :volume_24h_usdc, :liquidity_usdc,
                :yes_token_id, :no_token_id, :yes_price, :no_price, :raw_json,
                datetime('now')
            )
            ON CONFLICT(condition_id) DO UPDATE SET
                yes_price           = excluded.yes_price,
                no_price            = excluded.no_price,
                volume_24h_usdc     = CASE WHEN excluded.volume_24h_usdc > 0 THEN excluded.volume_24h_usdc ELSE markets.volume_24h_usdc END,
                liquidity_usdc      = CASE WHEN excluded.liquidity_usdc > 0 THEN excluded.liquidity_usdc ELSE markets.liquidity_usdc END,
                active              = excluded.active,
                closed              = excluded.closed,
                days_to_resolution  = excluded.days_to_resolution,
                last_updated        = datetime('now')
        """, market)


def get_active_markets(min_liquidity: float = 0, min_volume: float = 0,
                        max_days: int = 365, category: str = None) -> list:
    """Return active, non-closed markets matching filters."""
    query = """
        SELECT * FROM markets
        WHERE active = 1 AND closed = 0
          AND liquidity_usdc >= ?
          AND volume_24h_usdc >= ?
          AND days_to_resolution > 0
          AND days_to_resolution <= ?
    """
    params = [min_liquidity, min_volume, max_days]
    if category:
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY volume_24h_usdc DESC"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def get_market(condition_id: str) -> dict | None:
    """Return a single market by condition_id, or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM markets WHERE condition_id = ?", (condition_id,)
        ).fetchone()
        return dict(row) if row else None


# ─── SENTIMENT HELPERS ───────────────────────────────────────────────────────

def save_sentiment(condition_id: str, source: str, score: float,
                   post_count: int = 0, keywords: list = None,
                   raw_summary: str = None):
    """Persist a sentiment score for a market + source."""
    kw_json = json.dumps(keywords) if keywords else None
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO sentiment (condition_id, source, score, post_count,
                                   keywords, raw_summary)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (condition_id, source, score, post_count, kw_json, raw_summary))


def get_latest_sentiment(condition_id: str) -> list[dict]:
    """Return the most recent sentiment row per source for a market."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT s1.* FROM sentiment s1
            INNER JOIN (
                SELECT source, MAX(scraped_at) as max_at
                FROM sentiment WHERE condition_id = ?
                GROUP BY source
            ) s2 ON s1.source = s2.source AND s1.scraped_at = s2.max_at
            WHERE s1.condition_id = ?
        """, (condition_id, condition_id)).fetchall()
        return [dict(r) for r in rows]


# ─── PREDICTION HELPERS ─────────────────────────────────────────────────────

def save_prediction(pred: dict) -> int:
    """Persist a prediction dict and return its row ID."""
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO predictions
                (condition_id, xgboost_prob, sentiment_adj_prob, claude_prob,
                 market_yes_price, edge_pct, confidence, claude_reasoning)
            VALUES
                (:condition_id, :xgboost_prob, :sentiment_adj_prob, :claude_prob,
                 :market_yes_price, :edge_pct, :confidence, :claude_reasoning)
        """, pred)
        return cur.lastrowid


def get_latest_prediction(condition_id: str) -> dict | None:
    """Return the most recent prediction for a market."""
    with get_conn() as conn:
        row = conn.execute("""
            SELECT * FROM predictions
            WHERE condition_id = ?
            ORDER BY predicted_at DESC LIMIT 1
        """, (condition_id,)).fetchone()
        return dict(row) if row else None


# ─── TRADE HELPERS ───────────────────────────────────────────────────────────

def save_trade(trade: dict) -> int:
    """Persist a trade dict and return its row ID."""
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO trades
                (condition_id, prediction_id, direction, size_usdc,
                 kelly_fraction_used, bankroll_at_trade, status)
            VALUES
                (:condition_id, :prediction_id, :direction, :size_usdc,
                 :kelly_fraction_used, :bankroll_at_trade, 'pending')
        """, trade)
        return cur.lastrowid


def update_trade(trade_id: int, **kwargs):
    """Update trade fields by ID. Pass column=value kwargs."""
    if not kwargs:
        return
    allowed = {"order_id", "status", "avg_fill_price", "pnl_usdc",
               "settled_at", "placed_at"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    if not filtered:
        return
    fields = ", ".join(f"{k} = ?" for k in filtered)
    values = list(filtered.values()) + [trade_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE trades SET {fields} WHERE id = ?", values)


def get_open_trades() -> list[dict]:
    """Return all trades with status 'pending' or 'filled'."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status IN ('pending', 'filled')"
        ).fetchall()
        return [dict(r) for r in rows]


def get_trades_for_market(condition_id: str) -> list[dict]:
    """Return all trades for a given market."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE condition_id = ?", (condition_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# ─── AGGREGATE QUERIES ──────────────────────────────────────────────────────

def get_daily_pnl() -> float:
    """Sum of pnl_usdc for trades settled today."""
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COALESCE(SUM(pnl_usdc), 0) as daily_pnl
            FROM trades
            WHERE date(settled_at) = date('now')
              AND status = 'filled'
        """).fetchone()
        return row["daily_pnl"] if row else 0.0


def get_open_exposure() -> float:
    """Total USDC committed to open (pending + filled) trades."""
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COALESCE(SUM(size_usdc), 0) as exposure
            FROM trades WHERE status IN ('pending', 'filled') AND settled_at IS NULL
        """).fetchone()
        return row["exposure"] if row else 0.0


# ─── LEARNING HELPERS ────────────────────────────────────────────────────────

def save_learning(learning: dict):
    """Persist a post-mortem learning dict."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO learnings
                (trade_id, condition_id, outcome, pnl_usdc, our_prob,
                 market_prob, resolution, error_analysis, feature_flags)
            VALUES
                (:trade_id, :condition_id, :outcome, :pnl_usdc, :our_prob,
                 :market_prob, :resolution, :error_analysis, :feature_flags)
        """, learning)


def get_unreviewed_settled_trades() -> list[dict]:
    """Return settled/filled trades that have no learning entry yet."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT t.* FROM trades t
            LEFT JOIN learnings l ON t.id = l.trade_id
            WHERE t.status = 'filled' AND t.settled_at IS NOT NULL
              AND l.id IS NULL
        """).fetchall()
        return [dict(r) for r in rows]


# ─── BANKROLL ────────────────────────────────────────────────────────────────

def snapshot_bankroll(usdc_balance: float, open_positions: float):
    """Record a bankroll snapshot."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO bankroll (usdc_balance, open_positions, total_equity)
            VALUES (?, ?, ?)
        """, (usdc_balance, open_positions, usdc_balance + open_positions))


def get_performance_stats() -> dict:
    """Comprehensive performance metrics for status display."""
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

        resolved = conn.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN pnl_usdc > 0 THEN 1 ELSE 0 END),
                   SUM(pnl_usdc),
                   AVG(pnl_usdc)
            FROM trades
            WHERE status='filled' AND pnl_usdc IS NOT NULL
        """).fetchone()

        resolved_count = resolved[0] or 0
        wins = resolved[1] or 0
        win_rate = wins / resolved_count if resolved_count > 0 else 0

        best = conn.execute("""
            SELECT m.question, t.pnl_usdc FROM trades t
            JOIN markets m ON m.condition_id = t.condition_id
            WHERE t.pnl_usdc IS NOT NULL
            ORDER BY t.pnl_usdc DESC LIMIT 1
        """).fetchone()

        worst = conn.execute("""
            SELECT m.question, t.pnl_usdc FROM trades t
            JOIN markets m ON m.condition_id = t.condition_id
            WHERE t.pnl_usdc IS NOT NULL
            ORDER BY t.pnl_usdc ASC LIMIT 1
        """).fetchone()

        categories = conn.execute("""
            SELECT m.category,
                   COUNT(*) as trades,
                   SUM(t.pnl_usdc) as pnl
            FROM trades t
            JOIN markets m ON m.condition_id = t.condition_id
            WHERE t.pnl_usdc IS NOT NULL
            GROUP BY m.category
        """).fetchall()

        return {
            "total_trades": total,
            "resolved_trades": resolved_count,
            "win_rate": win_rate,
            "total_pnl": resolved[2] or 0,
            "avg_pnl_per_trade": resolved[3] or 0,
            "best_trade": dict(best) if best else None,
            "worst_trade": dict(worst) if worst else None,
            "category_breakdown": [dict(c) for c in categories],
        }
