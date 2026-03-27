"""
Prediction Bot - Configuration
Edit this file with your actual credentials before running.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# ─── API CREDENTIALS ──────────────────────────────────────────────────────────

POLYMARKET_API_KEY        = os.getenv("POLYMARKET_API_KEY", "")
POLYMARKET_API_SECRET     = os.getenv("POLYMARKET_API_SECRET", "")
POLYMARKET_API_PASSPHRASE = os.getenv("POLYMARKET_API_PASSPHRASE", "")
POLYMARKET_WALLET_ADDRESS = os.getenv("POLYMARKET_WALLET_ADDRESS", "")
POLYMARKET_PRIVATE_KEY    = os.getenv("POLYMARKET_PRIVATE_KEY", "")   # Polygon wallet pk

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

REDDIT_CLIENT_ID          = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET      = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT         = os.getenv("REDDIT_USER_AGENT", "prediction-bot/1.0")

# ─── POLYMARKET ENDPOINTS ─────────────────────────────────────────────────────

POLYMARKET_CLOB_URL   = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL  = "https://gamma-api.polymarket.com"   # market metadata
POLYMARKET_CHAIN_ID   = 137   # Polygon mainnet

# ─── CLAUDE MODEL ─────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-6"

# ─── RSS FEEDS ────────────────────────────────────────────────────────────────

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://feeds.reuters.com/reuters/topNews",
    "https://www.politico.com/rss/politicopicks.xml",
    "https://cryptonews.com/news/feed/",
    "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://feeds.npr.org/1001/rss.xml",
]

# ─── TRADING PARAMETERS ──────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Risk management parameters."""
    # Kelly Criterion
    kelly_fraction: float = 0.25        # fractional Kelly (0.25 = quarter Kelly - conservative)
    max_kelly_bet_pct: float = 0.10     # hard cap: never risk more than 10% of bankroll per trade

    # Exposure limits
    max_single_position_usdc: float = 15.0    # max $ per position
    max_total_exposure_usdc: float  = 100.0   # max $ across all open positions
    daily_loss_limit_usdc: float    = 25.0    # bot pauses if daily loss exceeds this

    # Duration-aware exposure split
    max_long_exposure_pct: float  = 0.40     # max 40% of exposure on markets > 3 days
    long_threshold_days: int      = 3        # markets > this are "long"

    # Edge filter - only trade if edge > threshold
    min_edge_pct: float = 0.06          # 6% minimum edge (our_prob - market_prob)
    min_liquidity_usdc: float = 2000.0  # minimum market liquidity to consider
    min_volume_24h_usdc: float = 1000.0 # minimum 24h volume
    max_days_to_resolution: int = 3     # focus on 1-3 day markets only

    # Sentiment weights for probability adjustment
    sentiment_weight: float = 0.15      # how much sentiment shifts XGBoost prediction

    # Bankroll
    bankroll_usdc: float = 97.47       # starting / current bankroll

    # Time horizon scoring weights (multiply opportunity_score)
    time_horizon_weights: dict = None

    def __post_init__(self):
        if self.time_horizon_weights is None:
            self.time_horizon_weights = {
                'same_day': 2.0,   # resolves within 24h
                'short':    1.5,   # 1-3 days
                'medium':   1.0,   # 3-7 days
                'long':     0.5,   # 7-30 days
            }


@dataclass
class FilterConfig:
    """Controls which markets get ingested and considered."""
    max_markets_to_scan: int = 300
    markets_per_page: int    = 100
    top_n_candidates: int    = 10

    # Categories to focus on (Polymarket tags)
    target_categories: list = None

    def __post_init__(self):
        """Initialize default categories if not provided."""
        if self.target_categories is None:
            self.target_categories = [
                "sports", "crypto",
                "economics", "politics",
            ]


RISK   = RiskConfig()
FILTER = FilterConfig()

# ─── DATABASE ─────────────────────────────────────────────────────────────────

DB_PATH = os.getenv("DB_PATH", "data/prediction_bot.db")

# ─── LOGGING ─────────────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"
LOG_FILE  = "logs/bot.log"
