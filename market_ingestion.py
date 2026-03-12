"""
data/market_ingestion.py — Polymarket data layer

Fetches all active markets from Polymarket CLOB + Gamma APIs,
normalizes them into a unified format, and persists to SQLite.

Endpoints used:
  CLOB  → real-time prices, order book liquidity
  Gamma → metadata (description, category, end_date, volume)
"""

import requests
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional
from config import (
    POLYMARKET_CLOB_URL, POLYMARKET_GAMMA_URL,
    FILTER, RISK
)
from data.db import upsert_market, init_db

logger = logging.getLogger(__name__)

# ─── RAW API FETCHERS ─────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, retries: int = 3) -> dict | list | None:
    """GET with retry + exponential backoff."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP {e.response.status_code} on {url} (attempt {attempt+1})")
            if e.response.status_code == 429:
                time.sleep(2 ** attempt)   # rate limit backoff
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on {url}: {e} (attempt {attempt+1})")
            time.sleep(2 ** attempt)
    logger.error(f"All retries failed for {url}")
    return None


def fetch_clob_markets(limit: int = 100, next_cursor: str = "") -> tuple[list, str]:
    """
    Fetch a page of markets from CLOB API.
    Returns (markets_list, next_cursor). next_cursor="" means last page.
    """
    params = {"limit": limit}
    if next_cursor:
        params["next_cursor"] = next_cursor

    data = _get(f"{POLYMARKET_CLOB_URL}/markets", params=params)
    if not data:
        return [], ""

    markets    = data.get("data", [])
    next_cur   = data.get("next_cursor", "")
    if next_cur == "LTE=":   # Polymarket's sentinel for "no more pages"
        next_cur = ""
    return markets, next_cur


def fetch_gamma_markets(offset: int = 0, limit: int = 100) -> list:
    """
    Fetch market metadata from Gamma API (richer metadata than CLOB).
    Gamma has: description, category, volume, end_date, etc.
    """
    params = {
        "limit": limit,
        "offset": offset,
        "active": "true",
        "closed": "false",
        "order": "volume24hr",
        "ascending": "false",
    }
    data = _get(f"{POLYMARKET_GAMMA_URL}/markets", params=params)
    return data if isinstance(data, list) else []


def fetch_market_orderbook(token_id: str) -> dict | None:
    """Fetch order book for a specific outcome token to get spread/depth."""
    return _get(f"{POLYMARKET_CLOB_URL}/book", params={"token_id": token_id})


# ─── NORMALIZER ──────────────────────────────────────────────────────────────

def _days_until(iso_date: Optional[str]) -> Optional[int]:
    """Return number of days from now until iso_date, or None."""
    if not iso_date:
        return None
    try:
        end = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return max(0, (end - now).days)
    except Exception:
        return None


def _extract_price(tokens: list, outcome: str) -> Optional[float]:
    """Extract price for YES or NO token from CLOB token list."""
    for t in tokens:
        if t.get("outcome", "").upper() == outcome.upper():
            price = t.get("price")
            return float(price) if price is not None else None
    return None


def _extract_token_id(tokens: list, outcome: str) -> Optional[str]:
    for t in tokens:
        if t.get("outcome", "").upper() == outcome.upper():
            return t.get("token_id")
    return None


def normalize_clob_market(raw: dict) -> dict | None:
    """
    Convert raw CLOB market dict → normalized market dict.
    Returns None if market should be skipped (non-binary, no price, etc.)
    """
    condition_id = raw.get("condition_id")
    if not condition_id:
        return None

    tokens = raw.get("tokens", [])
    if len(tokens) < 2:
        return None  # skip non-binary markets

    yes_price = _extract_price(tokens, "YES")
    no_price  = _extract_price(tokens, "NO")

    # Sanity: prices should sum to ~1.0 (within 5% for spread)
    if yes_price is None or no_price is None:
        return None
    if not (0.85 <= yes_price + no_price <= 1.15):
        return None

    end_date = raw.get("end_date_iso") or raw.get("game_start_time")
    days     = _days_until(end_date)

    return {
        "condition_id":       condition_id,
        "question":           raw.get("question", ""),
        "category":           raw.get("category", ""),
        "end_date_iso":       end_date,
        "days_to_resolution": days,
        "active":             1 if raw.get("active", True) else 0,
        "closed":             1 if raw.get("closed", False) else 0,
        "volume_24h_usdc":    float(raw.get("volume_24h", 0) or 0),
        "liquidity_usdc":     float(raw.get("liquidity", 0) or 0),
        "yes_token_id":       _extract_token_id(tokens, "YES"),
        "no_token_id":        _extract_token_id(tokens, "NO"),
        "yes_price":          yes_price,
        "no_price":           no_price,
        "raw_json":           json.dumps(raw),
    }


def merge_gamma_metadata(market: dict, gamma_data: dict) -> dict:
    """
    Enrich a normalized CLOB market with Gamma metadata.
    Gamma has better volume and category data.
    """
    if not gamma_data:
        return market

    # Prefer Gamma's volume (more accurate)
    gamma_vol = float(gamma_data.get("volume24hr", 0) or 0)
    if gamma_vol > 0:
        market["volume_24h_usdc"] = gamma_vol

    # Gamma has cleaner category tags
    category = gamma_data.get("category") or gamma_data.get("tags", [{}])
    if isinstance(category, list) and category:
        market["category"] = category[0].get("slug", market["category"])
    elif isinstance(category, str):
        market["category"] = category

    return market


# ─── FILTER ──────────────────────────────────────────────────────────────────

def passes_filter(market: dict) -> bool:
    """
    Apply RiskConfig + FilterConfig thresholds.
    Returns True if market is worth tracking.
    """
    if market["active"] == 0 or market["closed"] == 1:
        return False

    if market["liquidity_usdc"] < RISK.min_liquidity_usdc:
        return False

    if market["volume_24h_usdc"] < RISK.min_volume_24h_usdc:
        return False

    days = market.get("days_to_resolution")
    if days is not None and days > RISK.max_days_to_resolution:
        return False

    # Skip near-resolved markets (>95% or <5% — no edge)
    yes_p = market.get("yes_price", 0.5)
    if yes_p > 0.95 or yes_p < 0.05:
        return False

    return True


# ─── MAIN INGESTION LOOP ──────────────────────────────────────────────────────

def ingest_all_markets(max_markets: int = None) -> int:
    """
    Full ingestion: paginate CLOB API → normalize → filter → persist.
    Returns count of markets saved.
    """
    max_markets = max_markets or FILTER.max_markets_to_scan
    init_db()

    saved      = 0
    scanned    = 0
    next_cursor = ""

    logger.info(f"Starting market ingestion (target: {max_markets} markets)")

    while scanned < max_markets:
        page_size = min(FILTER.markets_per_page, max_markets - scanned)
        markets, next_cursor = fetch_clob_markets(limit=page_size, next_cursor=next_cursor)

        if not markets:
            logger.warning("Empty page received — stopping ingestion")
            break

        for raw in markets:
            scanned += 1
            normalized = normalize_clob_market(raw)
            if normalized is None:
                continue
            if passes_filter(normalized):
                upsert_market(normalized)
                saved += 1

        logger.info(f"Scanned {scanned} | Saved {saved} | cursor={next_cursor[:20] if next_cursor else 'END'}")

        if not next_cursor:
            break

        time.sleep(0.2)  # be polite to the API

    logger.info(f"Ingestion complete. {saved}/{scanned} markets passed filter.")
    return saved


def refresh_prices(condition_ids: list[str] = None) -> int:
    """
    Lightweight refresh: update YES/NO prices for already-saved markets.
    Used between full ingestions to keep prices current.
    """
    from data.db import get_active_markets, upsert_market

    markets = get_active_markets() if not condition_ids else \
              [{"condition_id": cid} for cid in condition_ids]

    updated = 0
    for m in markets:
        # Fetch single market from CLOB
        data = _get(f"{POLYMARKET_CLOB_URL}/markets/{m['condition_id']}")
        if not data:
            continue
        normalized = normalize_clob_market(data)
        if normalized:
            upsert_market(normalized)
            updated += 1
        time.sleep(0.1)

    logger.info(f"Price refresh: {updated} markets updated")
    return updated


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        refresh_prices()
    else:
        count = ingest_all_markets()
        print(f"\n✅ Ingested {count} markets into {__import__('config').DB_PATH}")
