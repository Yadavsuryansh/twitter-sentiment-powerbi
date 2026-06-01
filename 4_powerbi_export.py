"""
Twitter Sentiment Analysis — Step 4: Power BI Export
Generates structured CSV/Excel files and a data model ready for Power BI Desktop.

HOW TO USE IN POWER BI:
1. Run this script to generate all export files in dashboard/
2. Open Power BI Desktop → Get Data → Text/CSV (or Excel)
3. Import each table listed below
4. Build relationships as described in POWER_BI_GUIDE.md
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
os.makedirs("dashboard", exist_ok=True)


# ──────────────────────────────────────────────
# Helper: generate realistic sample data
# (Replace with: df = pd.read_csv("data/labeled_tweets.csv"))
# ──────────────────────────────────────────────
def generate_sample_data(n: int = 10_000) -> pd.DataFrame:
    """Generates realistic mock data matching the real pipeline output."""
    random.seed(42)
    np.random.seed(42)

    topics = ["AI", "Climate", "Sports", "Politics", "Crypto"]
    topic_weights = {"AI": 0.25, "Climate": 0.20, "Sports": 0.22, "Politics": 0.18, "Crypto": 0.15}
    sentiment_by_topic = {
        "AI":       {"positive": 0.54, "negative": 0.28, "neutral": 0.18},
        "Climate":  {"positive": 0.38, "negative": 0.46, "neutral": 0.16},
        "Sports":   {"positive": 0.62, "negative": 0.22, "neutral": 0.16},
        "Politics": {"positive": 0.30, "negative": 0.52, "neutral": 0.18},
        "Crypto":   {"positive": 0.49, "negative": 0.33, "neutral": 0.18},
    }

    records = []
    base_date = datetime(2025, 3, 1)

    for i in range(n):
        topic   = random.choices(topics, weights=[topic_weights[t] for t in topics])[0]
        weights = list(sentiment_by_topic[topic].values())
        sent    = random.choices(list(sentiment_by_topic[topic].keys()), weights=weights)[0]

        # Simulate compound score based on sentiment
        if sent == "positive":
            compound = round(random.uniform(0.05, 0.95), 4)
        elif sent == "negative":
            compound = round(random.uniform(-0.95, -0.05), 4)
        else:
            compound = round(random.uniform(-0.049, 0.049), 4)

        created_at = base_date + timedelta(
            days=random.randint(0, 55),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        confidence = round(random.uniform(0.55, 0.98), 4)

        records.append({
            "tweet_id":       i + 100_000,
            "topic":          topic,
            "sentiment":      sent,
            "vader_compound": compound,
            "confidence":     confidence,
            "retweet_count":  int(np.random.exponential(5)),
            "like_count":     int(np.random.exponential(12)),
            "word_count":     random.randint(5, 40),
            "created_at":     created_at,
            "hour_of_day":    created_at.hour,
            "day_of_week":    created_at.strftime("%A"),
            "date":           created_at.date(),
            "week":           created_at.strftime("%Y-W%U"),
            "month":          created_at.strftime("%Y-%m"),
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# Table 1 — Fact table: tweets
# ──────────────────────────────────────────────
def export_fact_tweets(df: pd.DataFrame):
    out = df[[
        "tweet_id", "topic", "sentiment", "vader_compound",
        "confidence", "retweet_count", "like_count",
        "word_count", "created_at", "date", "hour_of_day",
        "day_of_week", "week", "month"
    ]].copy()
    out["sentiment_score"] = out["vader_compound"]   # Alias for Power BI measures
    path = "dashboard/fact_tweets.csv"
    out.to_csv(path, index=False)
    log.info(f"[1/5] fact_tweets.csv  →  {len(out):,} rows")
    return out


# ──────────────────────────────────────────────
# Table 2 — Daily aggregates (for trend line chart)
# ──────────────────────────────────────────────
def export_daily_sentiment(df: pd.DataFrame):
    daily = df.groupby(["date", "topic", "sentiment"]).agg(
        tweet_count=("tweet_id", "count"),
        avg_vader=("vader_compound", "mean"),
        avg_confidence=("confidence", "mean"),
        total_likes=("like_count", "sum"),
        total_retweets=("retweet_count", "sum"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["pct"] = daily.groupby(["date", "topic"])["tweet_count"].transform(
        lambda x: (x / x.sum() * 100).round(2)
    )
    path = "dashboard/daily_sentiment.csv"
    daily.to_csv(path, index=False)
    log.info(f"[2/5] daily_sentiment.csv  →  {len(daily):,} rows")
    return daily


# ──────────────────────────────────────────────
# Table 3 — Topic summary KPIs (for card visuals)
# ──────────────────────────────────────────────
def export_topic_kpis(df: pd.DataFrame):
    total = df.groupby("topic").agg(
        total_tweets=("tweet_id", "count"),
        avg_vader=("vader_compound", "mean"),
        total_likes=("like_count", "sum"),
        total_retweets=("retweet_count", "sum"),
    ).reset_index()

    pivot = df.groupby(["topic", "sentiment"])["tweet_id"].count().unstack(fill_value=0).reset_index()
    pivot.columns.name = None
    for col in ["positive", "negative", "neutral"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["pct_positive"] = (pivot["positive"] / (pivot["positive"]+pivot["negative"]+pivot["neutral"]) * 100).round(2)
    pivot["pct_negative"] = (pivot["negative"] / (pivot["positive"]+pivot["negative"]+pivot["neutral"]) * 100).round(2)
    pivot["pct_neutral"]  = (pivot["neutral"]  / (pivot["positive"]+pivot["negative"]+pivot["neutral"]) * 100).round(2)

    kpis = total.merge(pivot[["topic","positive","negative","neutral","pct_positive","pct_negative","pct_neutral"]], on="topic")
    path = "dashboard/topic_kpis.csv"
    kpis.to_csv(path, index=False)
    log.info(f"[3/5] topic_kpis.csv  →  {len(kpis):,} rows")
    return kpis


# ──────────────────────────────────────────────
# Table 4 — Hourly heatmap data
# ──────────────────────────────────────────────
def export_hourly_heatmap(df: pd.DataFrame):
    heatmap = df.groupby(["day_of_week", "hour_of_day", "sentiment"]).agg(
        tweet_count=("tweet_id", "count")
    ).reset_index()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heatmap["day_num"] = heatmap["day_of_week"].map({d: i for i, d in enumerate(day_order)})
    path = "dashboard/hourly_heatmap.csv"
    heatmap.to_csv(path, index=False)
    log.info(f"[4/5] hourly_heatmap.csv  →  {len(heatmap):,} rows")
    return heatmap


# ──────────────────────────────────────────────
# Table 5 — Model metrics (for accuracy card)
# ──────────────────────────────────────────────
def export_model_metrics():
    metrics = pd.DataFrame([
        {"model": "Logistic Regression", "accuracy": 84.0, "precision": 83.5, "recall": 84.0, "f1": 83.7, "is_primary": True},
        {"model": "SVM (LinearSVC)",     "accuracy": 81.0, "precision": 80.8, "recall": 81.0, "f1": 80.9, "is_primary": False},
        {"model": "Naive Bayes",         "accuracy": 76.0, "precision": 75.2, "recall": 76.0, "f1": 75.6, "is_primary": False},
        {"model": "VADER only",          "accuracy": 71.0, "precision": 70.1, "recall": 71.0, "f1": 70.5, "is_primary": False},
    ])
    path = "dashboard/model_metrics.csv"
    metrics.to_csv(path, index=False)
    log.info(f"[5/5] model_metrics.csv  →  {len(metrics)} rows")
    return metrics


# ──────────────────────────────────────────────
# All-in-one Excel workbook (optional alternative)
# ──────────────────────────────────────────────
def export_excel_workbook(df, daily, kpis, heatmap, metrics):
    path = "dashboard/twitter_sentiment_powerbi.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df[[
            "tweet_id","topic","sentiment","vader_compound","confidence",
            "retweet_count","like_count","word_count","created_at","date",
            "hour_of_day","day_of_week","week","month"
        ]].to_excel(writer, sheet_name="Fact_Tweets",    index=False)
        daily.to_excel(writer,   sheet_name="Daily_Sentiment", index=False)
        kpis.to_excel(writer,    sheet_name="Topic_KPIs",      index=False)
        heatmap.to_excel(writer, sheet_name="Hourly_Heatmap",  index=False)
        metrics.to_excel(writer, sheet_name="Model_Metrics",   index=False)
    log.info(f"Excel workbook saved → {path}  (5 sheets)")


if __name__ == "__main__":
    log.info("Generating Power BI export data...")

    # ── Load real data (uncomment when running after full pipeline) ──
    # df = pd.read_csv("data/labeled_tweets.csv", parse_dates=["created_at"])

    # ── Use sample data for demo ──
    df = generate_sample_data(10_000)

    fact    = export_fact_tweets(df)
    daily   = export_daily_sentiment(df)
    kpis    = export_topic_kpis(df)
    heatmap = export_hourly_heatmap(df)
    metrics = export_model_metrics()
    export_excel_workbook(fact, daily, kpis, heatmap, metrics)

    log.info("\n✓ All files saved to dashboard/")
    log.info("  → Import twitter_sentiment_powerbi.xlsx into Power BI Desktop")
    log.info("  → Or connect individual CSVs via Get Data → Text/CSV")
