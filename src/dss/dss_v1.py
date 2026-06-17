import pandas as pd
import re
from pathlib import Path
from nltk.corpus import stopwords

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DSS_DIR = PROJECT_ROOT / "data" / "dss_samples"

STOP_WORDS = set(stopwords.words("english"))

# ==========================================
# Vocabulary
# ==========================================

def build_vocabulary(csv_path):

    df = pd.read_csv(csv_path)

    vocab = set()

    for text in df["text"]:

        words = re.findall(
            r"[a-z]+",
            str(text).lower()
        )

        words = [
            w for w in words
            if w not in STOP_WORDS
            and len(w) > 1
        ]

        vocab.update(words)

    return vocab

# ==========================================
# Metrics
# ==========================================

def jaccard(v1, v2):

    return len(
        v1 & v2
    ) / len(
        v1 | v2
    )

def oov(source, target):

    unseen = target - source

    return len(unseen) / len(target)

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    datasets = [
        "imdb",
        "twitter",
        "amazon",
        "yelp"
    ]

    vocabularies = {}

    for dataset in datasets:

        vocabularies[dataset] = build_vocabulary(
            DSS_DIR / f"{dataset}_dss.csv"
        )

    source = "imdb"

    print("\nDSS V1\n")

    for target in [
        "twitter",
        "amazon",
        "yelp"
    ]:

        j = jaccard(
            vocabularies[source],
            vocabularies[target]
        )

        o = oov(
            vocabularies[source],
            vocabularies[target]
        )

        dss = (
            0.5 * (1 - j)
            +
            0.5 * o
        )

        print(
            f"{source} -> {target}"
        )

        print(
            f"Jaccard : {j:.4f}"
        )

        print(
            f"OOV     : {o:.4f}"
        )

        print(
            f"DSS V1  : {dss:.4f}\n"
        )