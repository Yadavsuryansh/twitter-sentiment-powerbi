"""
Twitter Sentiment Analysis — Step 1: Data Collection
Uses Tweepy to stream tweets and store them in PostgreSQL.
"""

import tweepy
import psycopg2
import json
import logging
from datetime import datetime
from config import TWITTER_KEYS, DB_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# PostgreSQL: create table if not exists
# ──────────────────────────────────────────────
def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_tweets (
            id          BIGSERIAL PRIMARY KEY,
            tweet_id    TEXT UNIQUE,
            username    TEXT,
            text        TEXT,
            topic       TEXT,
            lang        TEXT,
            retweet_count INT DEFAULT 0,
            like_count  INT DEFAULT 0,
            created_at  TIMESTAMP,
            collected_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    log.info("Database initialised.")


# ──────────────────────────────────────────────
# Tweepy v2 StreamingClient
# ──────────────────────────────────────────────
class SentimentStreamClient(tweepy.StreamingClient):
    """Stream tweets in real time and persist to PostgreSQL."""

    TOPIC_KEYWORDS = {
        "AI":       ["ChatGPT", "OpenAI", "LLM", "artificial intelligence", "machine learning"],
        "Climate":  ["climate change", "global warming", "COP30", "carbon emissions", "renewable energy"],
        "Sports":   ["Champions League", "NBA", "NFL", "transfer", "goal"],
        "Politics": ["election", "senate", "policy", "government", "vote"],
        "Crypto":   ["Bitcoin", "Ethereum", "DeFi", "crypto", "blockchain"],
    }

    def __init__(self, bearer_token, **kwargs):
        super().__init__(bearer_token, **kwargs)
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.tweet_count = 0
        self.TARGET = 10_000

    def _detect_topic(self, text: str) -> str:
        text_lower = text.lower()
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(kw.lower() in text_lower for kw in keywords):
                return topic
        return "General"

    def on_tweet(self, tweet):
        if self.tweet_count >= self.TARGET:
            self.disconnect()
            return

        # Skip retweets
        if tweet.text.startswith("RT "):
            return

        topic = self._detect_topic(tweet.text)
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO raw_tweets (tweet_id, username, text, topic, lang, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tweet_id) DO NOTHING;
            """, (
                str(tweet.id),
                tweet.author_id,
                tweet.text,
                topic,
                getattr(tweet, "lang", "en"),
                tweet.created_at or datetime.utcnow(),
            ))
            self.conn.commit()
            self.tweet_count += 1
            if self.tweet_count % 500 == 0:
                log.info(f"Collected {self.tweet_count} tweets...")
        except Exception as e:
            log.error(f"DB insert error: {e}")
            self.conn.rollback()
        finally:
            cur.close()

    def on_errors(self, errors):
        log.error(f"Stream error: {errors}")

    def on_disconnect(self):
        log.info(f"Stream disconnected. Total collected: {self.tweet_count}")
        self.conn.close()


def add_stream_rules(client):
    """Add keyword-based rules to the filtered stream."""
    # Remove existing rules first
    existing = client.get_rules()
    if existing.data:
        ids = [r.id for r in existing.data]
        client.delete_rules(ids)

    rules = [
        tweepy.StreamRule("ChatGPT OR OpenAI OR LLM lang:en -is:retweet",   tag="AI"),
        tweepy.StreamRule("climate change OR COP30 lang:en -is:retweet",     tag="Climate"),
        tweepy.StreamRule("Champions League OR NBA lang:en -is:retweet",     tag="Sports"),
        tweepy.StreamRule("election OR senate lang:en -is:retweet",          tag="Politics"),
        tweepy.StreamRule("Bitcoin OR Ethereum OR DeFi lang:en -is:retweet", tag="Crypto"),
    ]
    client.add_rules(rules)
    log.info("Stream rules added.")


def collect_tweets():
    init_db()
    client = SentimentStreamClient(
        bearer_token=TWITTER_KEYS["bearer_token"],
        wait_on_rate_limit=True,
    )
    add_stream_rules(client)
    log.info("Starting tweet stream (target: 10,000 tweets)...")
    client.filter(
        tweet_fields=["created_at", "lang", "public_metrics", "author_id"],
        expansions=["author_id"],
    )


if __name__ == "__main__":
    collect_tweets()
