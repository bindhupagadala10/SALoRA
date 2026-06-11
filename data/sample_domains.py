import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

PROCESSED_DIR = BASE_DIR / "processed"
BALANCED_DIR = BASE_DIR / "balanced"

BALANCED_DIR.mkdir(exist_ok=True)

SAMPLE_SIZE = 4000
RANDOM_STATE = 42


def balance_dataset(filename):

    print(f"\nProcessing {filename}")

    df = pd.read_csv(
        PROCESSED_DIR / filename
    )

    if len(df) < SAMPLE_SIZE:
        raise ValueError(
            f"{filename} has only {len(df)} rows"
        )

    df_sample = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    output_name = filename.replace(
        ".csv",
        "_balanced.csv"
    )

    df_sample.to_csv(
        BALANCED_DIR / output_name,
        index=False
    )

    print(
        f"Saved {output_name} "
        f"({len(df_sample)} rows)"
    )


if __name__ == "__main__":

    balance_dataset("imdb.csv")
    balance_dataset("twitter.csv")
    balance_dataset("amazon.csv")
    balance_dataset("yelp.csv")

    print("\nDone!")