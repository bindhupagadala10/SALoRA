import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "processed"

DATASETS = [
    "imdb.csv",
    "twitter.csv",
    "amazon.csv",
    "yelp.csv"
]


def audit_dataset(filename):

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    df = pd.read_csv(PROCESSED_DIR / filename)

    total_rows = len(df)

    null_texts = df["text"].isna().sum()

    empty_texts = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    duplicate_texts = df.duplicated(
        subset=["text"]
    ).sum()

    positive_count = (
        df["label"] == 1
    ).sum()

    negative_count = (
        df["label"] == 0
    ).sum()

    cleaned_df = df.copy()

    cleaned_df = cleaned_df.dropna(
        subset=["text"]
    )

    cleaned_df["text"] = (
        cleaned_df["text"]
        .astype(str)
        .str.strip()
    )

    cleaned_df = cleaned_df[
        cleaned_df["text"] != ""
    ]

    cleaned_df = cleaned_df.drop_duplicates(
        subset=["text"]
    )

    final_rows = len(cleaned_df)

    print(f"Total Rows        : {total_rows:,}")
    print(f"Null Texts        : {null_texts:,}")
    print(f"Empty Texts       : {empty_texts:,}")
    print(f"Duplicate Texts   : {duplicate_texts:,}")
    print(f"Positive Labels   : {positive_count:,}")
    print(f"Negative Labels   : {negative_count:,}")
    print(f"Usable Rows       : {final_rows:,}")

    return final_rows


if __name__ == "__main__":

    usable_sizes = {}

    print("\nDATASET AUDIT REPORT")

    for dataset in DATASETS:

        usable_rows = audit_dataset(dataset)

        usable_sizes[
            dataset.replace(".csv", "")
        ] = usable_rows

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, size in usable_sizes.items():

        print(
            f"{name:<10} : {size:,}"
        )

    bottleneck = min(
        usable_sizes,
        key=usable_sizes.get
    )

    print("\n" + "=" * 70)
    print("BOTTLENECK DATASET")
    print("=" * 70)

    print(
        f"{bottleneck} -> "
        f"{usable_sizes[bottleneck]:,} rows"
    )