import pandas as pd

df = pd.read_csv("processed/amazon.csv")

print(df["label"].value_counts())