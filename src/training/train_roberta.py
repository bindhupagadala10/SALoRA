import json
import os
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "roberta-base"

DATA_PATH = "data/final/imdb_final.csv"

MODEL_OUTPUT_DIR = "models/roberta_imdb"

METRICS_OUTPUT_DIR = "results/metrics"

METRICS_OUTPUT_PATH = (
    f"{METRICS_OUTPUT_DIR}/roberta_imdb_metrics.json"
)

SEED = 42

# =====================================================
# SEED
# =====================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =====================================================
# DATA
# =====================================================

from sklearn.model_selection import train_test_split

def load_dataset(csv_path):

    df = pd.read_csv(csv_path)

    df = df[["text", "label"]].dropna()

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["label"]
    )

    return DatasetDict({
        "train": Dataset.from_pandas(
            train_df.reset_index(drop=True),
            preserve_index=False
        ),
        "validation": Dataset.from_pandas(
            val_df.reset_index(drop=True),
            preserve_index=False
        ),
        "test": Dataset.from_pandas(
            test_df.reset_index(drop=True),
            preserve_index=False
        )
    })# =====================================================
# METRICS
# =====================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    return {

        "accuracy":
        accuracy_score(
            labels,
            predictions
        ),

        "macro_f1":
        f1_score(
            labels,
            predictions,
            average="macro"
        )

    }

# =====================================================
# MAIN
# =====================================================

def main():

    set_seed(SEED)

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading dataset...")

    dataset = load_dataset(
        DATA_PATH
    )

    print(dataset)

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256
        )

    print("Tokenizing...")

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "negative",
            1: "positive"
        },
        label2id={
            "negative": 0,
            "positive": 1
        }
    )

    training_args = TrainingArguments(

        output_dir=MODEL_OUTPUT_DIR,

        eval_strategy="epoch",

        save_strategy="epoch",

        learning_rate=2e-5,

        per_device_train_batch_size=8,

        per_device_eval_batch_size=8,

        num_train_epochs=3,

        weight_decay=0.01,

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",

        greater_is_better=True,

        save_total_limit=1,

        seed=SEED,

        report_to="none"

    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=tokenized_dataset["train"],

        eval_dataset=tokenized_dataset["validation"],

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics

    )

    print("\nStarting training...\n")

    trainer.train()

    print("\nEvaluating...\n")

    validation_metrics = trainer.evaluate(
        tokenized_dataset["validation"],
        metric_key_prefix="validation"
    )

    test_metrics = trainer.evaluate(
        tokenized_dataset["test"],
        metric_key_prefix="test"
    )

    print("\nSaving model...")

    trainer.save_model(
        MODEL_OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        MODEL_OUTPUT_DIR
    )

    os.makedirs(
        METRICS_OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        METRICS_OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "validation":
                validation_metrics,

                "test":
                test_metrics
            },
            f,
            indent=2
        )

    print("\nDone!")

    print(
        f"\nMetrics saved to:\n{METRICS_OUTPUT_PATH}"
    )

# =====================================================

if __name__ == "__main__":

    main()