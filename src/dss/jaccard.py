import pandas as pd
import re
from pathlib import Path
from nltk.corpus import stopwords
from itertools import combinations

# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DSS_DIR = PROJECT_ROOT / "data" / "dss_samples"

# ==================================================
# Stopwords
# ==================================================

STOP_WORDS = set(stopwords.words("english"))

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
# Jaccard Similarity
# ==================================================

def jaccard_similarity(vocab_a, vocab_b):

    intersection = len(
        vocab_a.intersection(vocab_b)
    )

    union = len(
        vocab_a.union(vocab_b)
    )

    return intersection / union

# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    datasets = [
        "imdb",
        "twitter",
        "amazon",
        "yelp"
    ]

    vocabularies = {}

    print("\nBuilding vocabularies...\n")

    for dataset in datasets:

        vocabularies[dataset] = build_vocabulary(
            DSS_DIR / f"{dataset}_dss.csv"
        )

        print(
            f"{dataset:<10} "
            f"{len(vocabularies[dataset]):,}"
        )

    print("\n" + "=" * 60)
    print("JACCARD SIMILARITY")
    print("=" * 60)

    for a, b in combinations(
        datasets,
        2
    ):

        score = jaccard_similarity(
            vocabularies[a],
            vocabularies[b]
        )

        print(
            f"{a:<10} <-> {b:<10} : "
            f"{score:.4f}"
        )