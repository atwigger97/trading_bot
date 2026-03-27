"""
agents/research_agent.py — Sentiment research across Google News, Reddit, and RSS.

For each market, searches relevant content from three sources,
scores each using a VADER + FinBERT ensemble, and persists results
to the sentiment table via db.save_sentiment().

Public API (called by run_bot._process_market):
    get_sentiment(market) → {"news": float, "reddit": float,
                              "rss": float, "composite": float}

Each score in range [-1.0, +1.0].
"""

import re
import json
import logging
import time
from typing import Optional

import requests
import feedparser
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline as hf_pipeline

from config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    RSS_FEEDS,
)
from data.db import save_sentiment

logger = logging.getLogger(__name__)

# ─── SENTIMENT MODELS (lazy-loaded singletons) ──────────────────────────────

_vader: Optional[SentimentIntensityAnalyzer] = None
_finbert = None


def _get_vader() -> SentimentIntensityAnalyzer:
    """Lazy-load VADER analyzer."""
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
        logger.debug("VADER initialized")
    return _vader


def _get_finbert():
    """Lazy-load FinBERT pipeline. Returns None if model unavailable."""
    global _finbert
    if _finbert is None:
        try:
            _finbert = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
            logger.debug("FinBERT initialized")
        except Exception as e:
            logger.warning(f"FinBERT unavailable, VADER-only mode: {e}")
            _finbert = False  # sentinel: tried and failed
    return _finbert if _finbert is not False else None


# ─── SCORING ─────────────────────────────────────────────────────────────────

def _vader_score(text: str) -> float:
    """Return VADER compound score in [-1.0, +1.0]."""
    return _get_vader().polarity_scores(text)["compound"]


def _finbert_score(text: str) -> Optional[float]:
    """Return FinBERT score mapped to [-1.0, +1.0], or None."""
    model = _get_finbert()
    if model is None:
        return None
    try:
        result = model(text[:512])[0]
        label = result["label"].lower()
        conf = result["score"]
        if label == "positive":
            return conf
        elif label == "negative":
            return -conf
        return 0.0
    except Exception as e:
        logger.debug(f"FinBERT scoring error: {e}")
        return None


def _score_text(text: str) -> float:
    """Ensemble: average of VADER and FinBERT (if available)."""
    vader = _vader_score(text)
    finbert = _finbert_score(text)
    if finbert is not None:
        return (vader + finbert) / 2.0
    return vader


def _extract_keywords(question: str) -> list[str]:
    """Extract meaningful search keywords from a market question."""
    stop_words = {
        "will", "the", "be", "is", "are", "was", "were", "a", "an", "of",
        "in", "to", "for", "on", "by", "at", "or", "and", "not", "with",
        "this", "that", "from", "has", "have", "had", "do", "does", "did",
        "it", "its", "who", "what", "when", "where", "how", "which", "than",
        "but", "if", "then", "so", "no", "yes", "before", "after",
    }
    tokens = re.findall(r"[A-Za-z0-9]+", question)
    return [t for t in tokens if len(t) >= 3 and t.lower() not in stop_words][:8]


# ─── GOOGLE NEWS ─────────────────────────────────────────────────────────────

def _search_google_news(query: str, max_results: int = 15) -> list[str]:
    """
    Search Google News RSS for recent articles matching query.
    Returns list of 'title. summary' strings.
    """
    from urllib.parse import quote_plus
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        texts = []
        for entry in feed.entries[:max_results]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")[:300]
            texts.append(f"{title}. {summary}".strip())
        return texts
    except Exception as e:
        logger.warning(f"Google News search failed: {e}")
        return []


# ─── REDDIT ──────────────────────────────────────────────────────────────────

_reddit_client: Optional[praw.Reddit] = None

_CATEGORY_SUBREDDITS = {
    "politics":      ["politics", "geopolitics", "worldnews"],
    "crypto":        ["cryptocurrency", "bitcoin", "ethereum"],
    "sports":        ["sportsbook", "sports"],
    "finance":       ["wallstreetbets", "stocks", "economics"],
    "economics":     ["economics", "finance"],
    "entertainment": ["entertainment", "movies", "television"],
    "science":       ["science", "technology"],
}


def _get_reddit() -> Optional[praw.Reddit]:
    """Lazy-initialize Reddit client via PRAW."""
    global _reddit_client
    if _reddit_client is not None:
        return _reddit_client
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.debug("Reddit: no credentials, skipping")
        return None
    try:
        _reddit_client = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
        return _reddit_client
    except Exception as e:
        logger.warning(f"Reddit init failed: {e}")
        return None


def _search_reddit(query: str, category: str = "",
                   max_results: int = 15) -> list[str]:
    """Search Reddit for posts matching query in relevant subreddits."""
    reddit = _get_reddit()
    if reddit is None:
        return []

    cat_lower = category.lower().strip() if category else ""
    sub_names = _CATEGORY_SUBREDDITS.get(cat_lower, ["news", "worldnews"])

    texts = []
    try:
        subreddit = reddit.subreddit("+".join(sub_names))
        for submission in subreddit.search(query, sort="relevance",
                                           time_filter="week",
                                           limit=max_results):
            title = submission.title or ""
            body = (submission.selftext or "")[:300]
            texts.append(f"{title}. {body}".strip())
    except Exception as e:
        logger.warning(f"Reddit search failed: {e}")

    return texts


# ─── RSS ─────────────────────────────────────────────────────────────────────

def _search_rss(keywords: list[str], max_entries: int = 20) -> list[str]:
    """Fetch RSS feeds and return entries whose titles match any keyword."""
    if not keywords:
        return []

    pattern = re.compile("|".join(re.escape(kw) for kw in keywords), re.IGNORECASE)
    matched = []

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:50]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")[:300]
                combined = f"{title}. {summary}"
                if pattern.search(combined):
                    matched.append(combined)
                    if len(matched) >= max_entries:
                        return matched
        except Exception as e:
            logger.debug(f"RSS error for {feed_url}: {e}")

    return matched


# ─── AGGREGATE ───────────────────────────────────────────────────────────────

def _avg_score(texts: list[str]) -> float:
    """Average sentiment of a list of texts. Returns 0.0 if empty."""
    if not texts:
        return 0.0
    scores = [_score_text(t) for t in texts]
    return sum(scores) / len(scores)


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

def get_sentiment(market: dict) -> dict:
    """
    Run sentiment research for a single market across all sources.

    Args:
        market: Dict with at least 'condition_id', 'question', 'category'.

    Returns:
        {"twitter": float, "reddit": float, "rss": float, "composite": float}
    """
    condition_id = market["condition_id"]
    question = market.get("question", "")
    category = market.get("category", "")

    keywords = _extract_keywords(question)
    query_str = " ".join(keywords[:5])
    logger.debug(f"[{condition_id[:8]}] Research keywords: {keywords}")

    # ── Fetch texts ───────────────────────────────────────────────────────
    news_texts    = _search_google_news(query_str)
    reddit_texts  = _search_reddit(query_str, category=category)
    rss_texts     = _search_rss(keywords)

    # ── Score each source ─────────────────────────────────────────────────
    news_score    = _avg_score(news_texts)
    reddit_score  = _avg_score(reddit_texts)
    rss_score     = _avg_score(rss_texts)

    # ── Weighted composite (only from sources that returned data) ─────────
    weights = {"news": 0.45, "reddit": 0.30, "rss": 0.25}
    active = {}
    if news_texts:
        active["news"] = news_score
    if reddit_texts:
        active["reddit"] = reddit_score
    if rss_texts:
        active["rss"] = rss_score

    if active:
        total_w = sum(weights[k] for k in active)
        composite = sum(weights[k] * v / total_w for k, v in active.items())
    else:
        composite = 0.0

    composite = max(-1.0, min(1.0, composite))

    result = {
        "news":      round(news_score, 4),
        "reddit":    round(reddit_score, 4),
        "rss":       round(rss_score, 4),
        "composite": round(composite, 4),
    }

    # ── Persist to sentiment table ────────────────────────────────────────
    for source, texts, score in [("news",   news_texts,   news_score),
                                  ("reddit", reddit_texts, reddit_score),
                                  ("rss",    rss_texts,    rss_score)]:
        try:
            save_sentiment(
                condition_id=condition_id,
                source=source,
                score=round(score, 4),
                post_count=len(texts),
                keywords=keywords,
            )
        except Exception as e:
            logger.error(f"[{condition_id[:8]}] Failed to save {source} sentiment: {e}")

    logger.info(
        f"[{condition_id[:8]}] Sentiment — "
        f"news={result['news']:+.3f}({len(news_texts)}) "
        f"rd={result['reddit']:+.3f}({len(reddit_texts)}) "
        f"rss={result['rss']:+.3f}({len(rss_texts)}) "
        f"→ composite={result['composite']:+.3f}"
    )

    return result
