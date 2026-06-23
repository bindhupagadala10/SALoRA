import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.metrics import accuracy_score, f1_score


# ==================================================
# CHANGE ONLY THIS LINE
# ==================================================

SOURCE_DOMAIN = "twitter"

# ==================================================

MODEL_PATH = f"models/roberta_{SOURCE_DOMAIN}"

ALL_DATASETS = {
    "imdb": "data/final/imdb_final.csv",
    "twitter": "data/final/twitter_final.csv",
    "amazon": "data/final/amazon_final.csv",
    "yelp": "data/final/yelp_final.csv",
}

print(f"\nLoading model: {SOURCE_DOMAIN}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    processing_class=tokenizer
)

results = []

for target_domain, csv_path in ALL_DATASETS.items():

    # skip self-domain evaluation
    if target_domain == SOURCE_DOMAIN:
        continue

    print(f"\nEvaluating {SOURCE_DOMAIN} -> {target_domain}")

    df = pd.read_csv(csv_path)

    df = df[["text", "label"]].dropna()

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    dataset = Dataset.from_pandas(
        df,
        preserve_index=False
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128
        )

    dataset = dataset.map(
        tokenize,
        batched=True
    )

    predictions = trainer.predict(dataset)

    preds = predictions.predictions.argmax(axis=1)

    accuracy = accuracy_score(
        df["label"],
        preds
    )

    macro_f1 = f1_score(
        df["label"],
        preds,
        average="macro"
    )

    results.append({
        "source": SOURCE_DOMAIN,
        "target": target_domain,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

results_df = pd.DataFrame(results)

print("\n==========================")
print("TRANSFER RESULTS")
print("==========================")
print(results_df)

output_path = (
    f"results/metrics/"
    f"{SOURCE_DOMAIN}_transfer_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print(f"\nSaved to: {output_path}")