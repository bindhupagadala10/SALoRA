import pandas as pd
import re
from pathlib import Path
from nltk.corpus import stopwords
from itertools import permutations

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DSS_DIR = PROJECT_ROOT / "data" / "dss_samples"

STOP_WORDS = set(stopwords.words("english"))

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


def oov_rate(source_vocab, target_vocab):

    unseen = target_vocab - source_vocab

    return len(unseen) / len(target_vocab)


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

    print("\nOOV RATES\n")

    for source, target in permutations(
        datasets,
        2
    ):

        score = oov_rate(
            vocabularies[source],
            vocabularies[target]
        )

        print(
            f"{source:<10} -> "
            f"{target:<10} : "
            f"{score:.4f}"
        )