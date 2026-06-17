import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DSS_DIR = PROJECT_ROOT / "data" / "dss_samples"

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

datasets = [
    "imdb",
    "twitter",
    "amazon",
    "yelp"
]

for dataset in datasets:

    print(f"\nProcessing {dataset}")

    df = pd.read_csv(
        DSS_DIR /
        f"{dataset}_dss.csv"
    )

    texts = (
        df["text"]
        .astype(str)
        .tolist()
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    print(
        embeddings.shape
    )