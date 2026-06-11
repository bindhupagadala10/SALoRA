import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

PROCESSED_DIR = BASE_DIR / "processed"
DSS_DIR = BASE_DIR / "dss_samples"

DSS_DIR.mkdir(exist_ok=True)

# Smallest usable dataset after audit
SAMPLE_SIZE = 8537

RANDOM_STATE = 42


def create_dss_sample(filename):

    print(f"\nProcessing {filename}")

    df = pd.read_csv(
        PROCESSED_DIR / filename
    )

    # Clean again to ensure consistency

    df = df.dropna(subset=["text"])

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"] != ""
    ]

    df = df.drop_duplicates(
        subset=["text"]
    )

    if len(df) < SAMPLE_SIZE:

        raise ValueError(
            f"{filename} only has "
            f"{len(df)} usable rows"
        )

    sample_df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    output_name = filename.replace(
        ".csv",
        "_dss.csv"
    )

    sample_df.to_csv(
        DSS_DIR / output_name,
        index=False
    )

    print(
        f"Saved {output_name} "
        f"({len(sample_df)} rows)"
    )


if __name__ == "__main__":

    create_dss_sample("imdb.csv")
    create_dss_sample("twitter.csv")
    create_dss_sample("amazon.csv")
    create_dss_sample("yelp.csv")

    print("\nDSS sampling complete.")