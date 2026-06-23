import pandas as pd

for f in [
    "data/balanced/imdb_balanced.csv",
    "data/balanced/twitter_balanced.csv",
    "data/balanced/amazon_balanced.csv",
    "data/balanced/yelp_balanced.csv"
]:
    df = pd.read_csv(f)

    print(f)
    print(len(df))
    print(df["label"].value_counts())
    print()