import pandas as pd
from pathlib import Path

# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).parent

RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

PROCESSED_DIR.mkdir(exist_ok=True)

# ==================================================
# IMDB
# ==================================================

def preprocess_imdb():

    print("\nProcessing IMDB...")

    df = pd.read_csv(
        RAW_DIR / "imdb" / "IMDB Dataset.csv"
    )

    df = df.rename(
        columns={
            "review": "text"
        }
    )

    df["label"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    df = df[["text", "label"]]

    df = df.dropna()

    df.to_csv(
        PROCESSED_DIR / "imdb.csv",
        index=False
    )

    print(f"IMDB saved: {len(df)} rows")


# ==================================================
# SENTIMENT140
# ==================================================

def preprocess_sentiment140():

    print("\nProcessing Sentiment140...")

    train_df = pd.read_csv(
        RAW_DIR / "sentiment140" / "train_data.csv"
    )

    test_df = pd.read_csv(
        RAW_DIR / "sentiment140" / "test_data.csv"
    )

    df = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    df = df.rename(
        columns={
            "sentence": "text",
            "sentiment": "label"
        }
    )

    df = df[["text", "label"]]

    df = df.dropna()

    df.to_csv(
        PROCESSED_DIR / "twitter.csv",
        index=False
    )

    print(f"Twitter saved: {len(df)} rows")


# ==================================================
# AMAZON
# ==================================================

def preprocess_amazon():

    print("\nProcessing Amazon...")

    df = pd.read_csv(
        RAW_DIR / "amazon" / "amazon_reviews.csv"
    )

    df = df[
        ["reviewText", "overall"]
    ]

    df = df.rename(
        columns={
            "reviewText": "text"
        }
    )

    df = df.dropna()

    # Remove neutral reviews
    df = df[
        df["overall"] != 3
    ]

    # 4,5 -> positive
    # 1,2 -> negative

    df["label"] = (
        df["overall"] >= 4
    ).astype(int)

    df = df[
        ["text", "label"]
    ]

    df.to_csv(
        PROCESSED_DIR / "amazon.csv",
        index=False
    )

    print(f"Amazon saved: {len(df)} rows")


# ==================================================
# YELP
# ==================================================

def preprocess_yelp():

    print("\nProcessing Yelp...")

    df = pd.read_csv(
        RAW_DIR / "yelp" / "yelp.csv"
    )

    df = df[
        ["text", "stars"]
    ]

    df = df.dropna()

    # Remove neutral reviews
    df = df[
        df["stars"] != 3
    ]

    # 4,5 -> positive
    # 1,2 -> negative

    df["label"] = (
        df["stars"] >= 4
    ).astype(int)

    df = df[
        ["text", "label"]
    ]

    df.to_csv(
        PROCESSED_DIR / "yelp.csv",
        index=False
    )

    print(f"Yelp saved: {len(df)} rows")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STARTING DATA PREPROCESSING")
    print("=" * 60)

    preprocess_imdb()
    preprocess_sentiment140()
    preprocess_amazon()
    preprocess_yelp()

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print("\nGenerated Files:")

    for file in PROCESSED_DIR.glob("*.csv"):
        print(" -", file.name)