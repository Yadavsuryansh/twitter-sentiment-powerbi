"""
Twitter Sentiment Analysis — Step 2: Text Preprocessing
Tokenization, stopword removal, lemmatization, TF-IDF feature extraction.
"""

import re
import string
import logging
import pandas as pd
import psycopg2
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pickle
import os

from config import DB_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Download required NLTK data (run once)
for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "never", "very", "but"}
TOKENIZER  = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
LEMMATIZER = WordNetLemmatizer()


# ──────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    """Remove URLs, mentions, hashtag symbols, special chars."""
    text = re.sub(r"http\S+|www\S+",   "", text)   # URLs
    text = re.sub(r"@\w+",             "", text)   # Mentions
    text = re.sub(r"#(\w+)",           r"\1", text)  # Hashtags → word only
    text = re.sub(r"&amp;|&lt;|&gt;",  " ", text)  # HTML entities
    text = re.sub(r"[^\w\s']",         " ", text)  # Special chars
    text = re.sub(r"\s+",              " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> str:
    """Tokenize → remove stopwords → lemmatize → rejoin."""
    tokens = TOKENIZER.tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess(text: str) -> str:
    return tokenize_and_lemmatize(clean_tweet(text))


# ──────────────────────────────────────────────
# Load from DB and preprocess
# ──────────────────────────────────────────────
def load_and_preprocess() -> pd.DataFrame:
    log.info("Loading tweets from PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT tweet_id, username, text, topic, lang, retweet_count, like_count, created_at
        FROM raw_tweets
        WHERE lang = 'en'
        ORDER BY created_at DESC;
    """, conn)
    conn.close()

    log.info(f"Loaded {len(df)} tweets. Starting preprocessing...")
    df["clean_text"]     = df["text"].apply(clean_tweet)
    df["processed_text"] = df["clean_text"].apply(tokenize_and_lemmatize)
    df["text_length"]    = df["clean_text"].apply(len)
    df["word_count"]     = df["processed_text"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= 3].reset_index(drop=True)  # Remove very short tweets

    log.info(f"Preprocessing complete. {len(df)} usable tweets.")
    return df


# ──────────────────────────────────────────────
# TF-IDF vectorisation
# ──────────────────────────────────────────────
def build_tfidf(df: pd.DataFrame, max_features: int = 5000):
    log.info("Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),        # Unigrams + bigrams
        min_df=3,                  # Ignore very rare terms
        max_df=0.90,               # Ignore very common terms
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["processed_text"])
    log.info(f"TF-IDF shape: {X.shape}")

    # Persist for reuse
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    log.info("TF-IDF vectorizer saved → models/tfidf_vectorizer.pkl")
    return X, vectorizer


if __name__ == "__main__":
    df = load_and_preprocess()
    X, vectorizer = build_tfidf(df)
    df.to_csv("data/preprocessed_tweets.csv", index=False)
    log.info("Preprocessed data saved → data/preprocessed_tweets.csv")
    print(df[["text", "clean_text", "processed_text"]].head(3).to_string())
