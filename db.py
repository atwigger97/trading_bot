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

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@contextmanager
def get_conn():
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
                yes_price        = excluded.yes_price,
                no_price         = excluded.no_price,
                volume_24h_usdc  = excluded.volume_24h_usdc,
                liquidity_usdc   = excluded.liquidity_usdc,
                active           = excluded.active,
                closed           = excluded.closed,
                last_updated     = datetime('now')
        """, market)


def get_active_markets(min_liquidity: float = 0, min_volume: float = 0,
                        max_days: int = 365, category: str = None) -> list:
    query = """
        SELECT * FROM markets
        WHERE active = 1 AND closed = 0
          AND liquidity_usdc >= ?
          AND volume_24h_usdc >= ?
          AND (days_to_resolution IS NULL OR days_to_resolution <= ?)
    """
    params = [min_liquidity, min_volume, max_days]
    if category:
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY volume_24h_usdc DESC"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def save_prediction(pred: dict) -> int:
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


def save_trade(trade: dict) -> int:
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
    fields = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [trade_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE trades SET {fields} WHERE id = ?", values)


def save_learning(learning: dict):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO learnings
                (trade_id, condition_id, outcome, pnl_usdc, our_prob,
                 market_prob, resolution, error_analysis, feature_flags)
            VALUES
                (:trade_id, :condition_id, :outcome, :pnl_usdc, :our_prob,
                 :market_prob, :resolution, :error_analysis, :feature_flags)
        """, learning)


def snapshot_bankroll(usdc_balance: float, open_positions: float):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO bankroll (usdc_balance, open_positions, total_equity)
            VALUES (?, ?, ?)
        """, (usdc_balance, open_positions, usdc_balance + open_positions))


def get_daily_pnl() -> float:
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COALESCE(SUM(pnl_usdc), 0) as daily_pnl
            FROM trades
            WHERE date(settled_at) = date('now')
              AND status = 'filled'
        """).fetchone()
        return row["daily_pnl"] if row else 0.0


def get_open_exposure() -> float:
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COALESCE(SUM(size_usdc), 0) as exposure
            FROM trades WHERE status IN ('pending', 'filled') AND settled_at IS NULL
        """).fetchone()
        return row["exposure"] if row else 0.0
