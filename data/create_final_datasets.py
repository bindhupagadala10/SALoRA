import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PROCESSED = BASE_DIR / "processed"
FINAL = BASE_DIR / "final"

FINAL.mkdir(exist_ok=True)

TARGET_PER_CLASS = 1676

datasets = [
    "imdb",
    "twitter",
    "amazon",
    "yelp"
]

for name in datasets:

    print(f"\nProcessing {name}...")

    df = pd.read_csv(
        PROCESSED / f"{name}.csv"
    )

    pos = df[
        df["label"] == 1
    ].sample(
        TARGET_PER_CLASS,
        random_state=42
    )

    neg = df[
        df["label"] == 0
    ].sample(
        TARGET_PER_CLASS,
        random_state=42
    )

    final_df = pd.concat(
        [pos, neg],
        ignore_index=True
    )

    final_df = final_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    output = FINAL / f"{name}_final.csv"

    final_df.to_csv(
        output,
        index=False
    )

    print(
        f"Saved {output.name}: "
        f"{len(final_df)} rows"
    )

print("\nDone!")