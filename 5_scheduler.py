"""
Twitter Sentiment Analysis — Step 5: Automated Daily Pipeline
Runs the full pipeline on a schedule: collect → preprocess → classify → export.
Uses Python's APScheduler for in-process scheduling.
"""

import logging
import traceback
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas as pd
import psycopg2
import pickle

from preprocessing import load_and_preprocess, build_tfidf
from modeling import apply_vader
from powerbi_export import (
    export_fact_tweets, export_daily_sentiment,
    export_topic_kpis, export_hourly_heatmap,
    export_model_metrics, export_excel_workbook,
)
from config import DB_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ]
)
log = logging.getLogger(__name__)
scheduler = BlockingScheduler(timezone="UTC")


# ──────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────
def run_pipeline():
    start = datetime.utcnow()
    log.info(f"{'='*50}")
    log.info(f"Pipeline run started: {start.isoformat()}")

    try:
        # 1. Preprocess new tweets from DB
        df = load_and_preprocess()
        if df.empty:
            log.warning("No new tweets found. Skipping run.")
            return

        # 2. Build/update TF-IDF
        X, vectorizer = build_tfidf(df)

        # 3. Apply VADER labels
        df = apply_vader(df)

        # 4. Classify with trained LR model
        with open("models/logistic_regression.pkl", "rb") as f:
            lr = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        X_scaled   = scaler.transform(X)
        df["sentiment"] = lr.predict(X_scaled)
        df["confidence"] = lr.predict_proba(X_scaled).max(axis=1).round(4)

        # Add time columns
        df["created_at"]  = pd.to_datetime(df["created_at"])
        df["date"]        = df["created_at"].dt.date
        df["hour_of_day"] = df["created_at"].dt.hour
        df["day_of_week"] = df["created_at"].dt.strftime("%A")
        df["week"]        = df["created_at"].dt.strftime("%Y-W%U")
        df["month"]       = df["created_at"].dt.strftime("%Y-%m")

        # 5. Export to Power BI tables
        fact    = export_fact_tweets(df)
        daily   = export_daily_sentiment(df)
        kpis    = export_topic_kpis(df)
        heatmap = export_hourly_heatmap(df)
        metrics = export_model_metrics()
        export_excel_workbook(fact, daily, kpis, heatmap, metrics)

        # 6. Log run to DB
        _log_run(start, len(df), success=True)

        elapsed = (datetime.utcnow() - start).total_seconds()
        log.info(f"Pipeline complete in {elapsed:.1f}s. Processed {len(df):,} tweets.")

    except Exception as e:
        log.error(f"Pipeline FAILED: {e}")
        traceback.print_exc()
        _log_run(start, 0, success=False, error=str(e))


def _log_run(started_at, tweet_count, success, error=None):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id          BIGSERIAL PRIMARY KEY,
                started_at  TIMESTAMP,
                finished_at TIMESTAMP DEFAULT NOW(),
                tweet_count INT,
                success     BOOLEAN,
                error_msg   TEXT
            );
        """)
        cur.execute("""
            INSERT INTO pipeline_runs (started_at, tweet_count, success, error_msg)
            VALUES (%s, %s, %s, %s);
        """, (started_at, tweet_count, success, error))
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        log.warning(f"Could not log run to DB: {e}")


# ──────────────────────────────────────────────
# Schedule: run at 03:00 UTC every day
# ──────────────────────────────────────────────
@scheduler.scheduled_job(CronTrigger(hour=3, minute=0))
def daily_job():
    log.info("Scheduled daily pipeline triggered.")
    run_pipeline()


if __name__ == "__main__":
    log.info("Starting scheduler (daily at 03:00 UTC). Press Ctrl+C to stop.")
    log.info("Running pipeline once immediately...")
    run_pipeline()          # Run immediately on start
    scheduler.start()       # Then schedule for 03:00 UTC daily
