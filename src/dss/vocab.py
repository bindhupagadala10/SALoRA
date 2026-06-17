import pandas as pd
import re
from pathlib import Path
from nltk.corpus import stopwords

# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = BASE_DIR.parent.parent

DSS_DIR = PROJECT_ROOT / "data" / "dss_samples"

# ==================================================
# Stopwords
# ==================================================

STOP_WORDS = set(
    stopwords.words("english")
)

# ==================================================
# Vocabulary Builder
# ==================================================

def build_vocabulary(csv_path):

    df = pd.read_csv(csv_path)

    vocabulary = set()

    for text in df["text"]:

        text = str(text).lower()

        words = re.findall(
            r"[a-z]+",
            text
        )

        words = [
            word
            for word in words
            if word not in STOP_WORDS
            and len(word) > 1
        ]

        vocabulary.update(words)

    return vocabulary

# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("VOCABULARY ANALYSIS")
    print("=" * 60)

    datasets = [
        "imdb",
        "twitter",
        "amazon",
        "yelp"
    ]

    vocabularies = {}

    for dataset in datasets:

        file_path = (
            DSS_DIR /
            f"{dataset}_dss.csv"
        )

        print(
            f"\nProcessing {dataset}..."
        )

        vocab = build_vocabulary(
            file_path
        )

        vocabularies[
            dataset
        ] = vocab

        print(
            f"Vocabulary Size: "
            f"{len(vocab):,}"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for dataset, vocab in vocabularies.items():

        print(
            f"{dataset:<10} "
            f"{len(vocab):,}"
        )

    print("\nDone!")