"""
scripts/run_bot.py — Main entrypoint

Run modes:
    python scripts/run_bot.py ingest      # one-time market ingestion
    python scripts/run_bot.py loop        # full trading loop (live)
    python scripts/run_bot.py dry-run     # full loop, no real orders placed
"""

import sys
import os
import logging
import schedule
import time
from dotenv import load_dotenv
load_dotenv()

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log"),
    ]
)
logger = logging.getLogger("run_bot")

from data.db import init_db, get_daily_pnl, get_open_exposure, get_open_trades
from data.market_ingestion import ingest_all_markets, refresh_prices
from config import RISK
from agents.notify_agent import (
    send_notification, notify_trade_placed, notify_trade_filled,
    notify_daily_loss_limit, notify_daily_summary,
)


def trading_cycle(dry_run: bool = False):
    """
    One full cycle: ingest → filter → predict → risk check → execute → learn.
    Runs every N minutes in the main loop.
    """
    from data.db import get_active_markets
    from config import RISK, FILTER

    logger.info("─── Starting trading cycle ───")

    # ── Reconcile any pending orders first ──
    try:
        from agents.execute_agent import reconcile_pending_orders, settle_resolved_positions
        reconcile_pending_orders()
        settle_resolved_positions()   # write pnl_usdc + settled_at for resolved markets
    except Exception as e:
        logger.warning(f"Order reconciliation failed: {e}")

    # ── Safety checks ──
    daily_pnl = get_daily_pnl()
    if daily_pnl <= -RISK.daily_loss_limit_usdc:
        logger.warning(f"Daily loss limit hit (${daily_pnl:.2f}). Pausing.")
        notify_daily_loss_limit(daily_pnl)
        return

    exposure = get_open_exposure()
    if exposure >= RISK.max_total_exposure_usdc:
        logger.warning(f"Max exposure reached (${exposure:.2f}). Skipping new trades.")
        return

    # ── Step 1: Refresh prices ──
    refresh_prices()

    # ── Step 2: Get filtered candidates ──
    candidates = get_active_markets(
        min_liquidity=RISK.min_liquidity_usdc,
        min_volume=RISK.min_volume_24h_usdc,
        max_days=RISK.max_days_to_resolution,
    )
    logger.info(f"Candidates for this cycle: {len(candidates)}")

    # Skip markets where we already have an open position
    open_condition_ids = {
        t["condition_id"] for t in get_open_trades()
    }
    candidates = [
        m for m in candidates
        if m["condition_id"] not in open_condition_ids
    ]
    logger.info(f"After de-dup: {len(candidates)} (skipped {len(open_condition_ids)} with open trades)")

    new_trades = 0
    max_new_per_cycle = 2  # at most 2 new trades per 5-min cycle
    for market in candidates[:5]:   # evaluate top 5, but cap new trades
        if new_trades >= max_new_per_cycle:
            logger.info(f"Hit per-cycle trade limit ({max_new_per_cycle}). Stopping.")
            break
        try:
            placed = _process_market(market, dry_run=dry_run)
            if placed:
                new_trades += 1
        except Exception as e:
            logger.error(f"Error processing {market['condition_id']}: {e}", exc_info=True)

    # ── Step 6: Learn from settled trades ──
    try:
        from agents.learn_agent import review_settled_trades
        reviewed = review_settled_trades()
        if reviewed:
            logger.info(f"Learn agent reviewed {reviewed} settled trades")
    except Exception as e:
        logger.error(f"Learn agent failed: {e}", exc_info=True)

    logger.info("─── Cycle complete ───")


def _process_market(market: dict, dry_run: bool = False) -> bool:
    """
    Run the full pipeline for a single market.
    Returns True if a trade was placed (or would be in dry-run).
    """
    from agents.research_agent import get_sentiment
    from agents.predict_agent import predict
    from agents.risk_agent import approve_trade
    from agents.execute_agent import place_order

    condition_id = market["condition_id"]

    # Research
    sentiment = get_sentiment(market)

    # Predict
    prediction = predict(market, sentiment)
    if prediction is None:
        return False

    edge = prediction["edge_pct"]
    confidence = prediction.get("confidence", "low")

    if abs(edge) < RISK.min_edge_pct:
        logger.debug(f"Insufficient edge ({edge:.1%}) on {condition_id[:8]}")
        return False

    if confidence == "low":
        logger.debug(f"Low confidence on {condition_id[:8]}, skipping")
        return False

    logger.info(f"Edge found: {edge:+.1%} conf={confidence} on '{market['question'][:60]}'")

    # Risk
    approved, size_usdc = approve_trade(prediction, market)
    if not approved or size_usdc <= 0:
        return False

    # Execute
    if dry_run:
        logger.info(f"[DRY RUN] Would place {prediction['direction']} "
                    f"${size_usdc:.2f} on {condition_id[:8]}")
    else:
        order_id = place_order(market, prediction, size_usdc)
        if order_id:
            notify_trade_placed(market, prediction, size_usdc)
    return True


def run_learn_agent():
    """Daily job: review settled trades for post-mortem analysis."""
    try:
        from agents.learn_agent import review_settled_trades
        reviewed = review_settled_trades()
        if reviewed:
            logger.info(f"Learn agent reviewed {reviewed} settled trades")
            send_notification(f"\U0001f4d6 Learn agent reviewed {reviewed} settled trades")
    except Exception as e:
        logger.error(f"Learn agent daily job failed: {e}")


def retrain_model():
    """Weekly job: retrain XGBoost model on accumulated data."""
    try:
        import sys
        sys.path.insert(0, "files")
        from xgboost_model import train
        metrics = train()
        auc = metrics.get("cv_auc_mean", 0)
        n = metrics.get("n_samples", 0)
        logger.info(f"Model retrained: AUC={auc:.3f} n={n}")
        send_notification(
            f"\U0001f9e0 Model retrained.\n"
            f"AUC={auc:.3f} | n={n}"
        )
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        send_notification(f"\u26a0\ufe0f Model retrain failed: {e}")


def main():
    """Entry point — parse mode from CLI args."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "loop"
    init_db()

    if mode == "ingest":
        count = ingest_all_markets()
        print(f"✅ Ingested {count} markets")

    elif mode == "dry-run":
        logger.info("Starting DRY RUN loop")
        ingest_all_markets()
        schedule.every(5).minutes.do(trading_cycle, dry_run=True)
        while True:
            schedule.run_pending()
            time.sleep(10)

    elif mode == "loop":
        logger.info("Starting LIVE trading loop ⚠️")
        ingest_all_markets()
        # Full ingestion every 6 hours, price refresh every 5 min
        schedule.every(6).hours.do(ingest_all_markets)
        schedule.every(5).minutes.do(trading_cycle, dry_run=False)
        # Daily jobs
        schedule.every().day.at("00:00").do(run_learn_agent)
        schedule.every().day.at("00:05").do(notify_daily_summary)
        # Weekly retrain
        schedule.every().sunday.at("02:00").do(retrain_model)
        while True:
            schedule.run_pending()
            time.sleep(10)

    else:
        print(f"Unknown mode: {mode}. Use: ingest | dry-run | loop")
        sys.exit(1)


if __name__ == "__main__":
    main()
