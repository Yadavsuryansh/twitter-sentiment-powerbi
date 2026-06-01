"""
config.py — Credentials and settings
Replace all placeholder values with your actual keys before running.
"""

# ── Twitter API v2 keys (from developer.twitter.com) ──
TWITTER_KEYS = {
    "bearer_token":        "YOUR_BEARER_TOKEN_HERE",
    "consumer_key":        "YOUR_API_KEY_HERE",
    "consumer_secret":     "YOUR_API_SECRET_HERE",
    "access_token":        "YOUR_ACCESS_TOKEN_HERE",
    "access_token_secret": "YOUR_ACCESS_TOKEN_SECRET_HERE",
}

# ── PostgreSQL connection ──
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "twitter_sentiment",
    "user":     "postgres",
    "password": "YOUR PASSWORD HERE",
}
