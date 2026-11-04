from datasets import load_dataset
import pandas as pd

ds = load_dataset("masakhane/masakhanews", "fra")
print(ds)  # splits et nombre de lignes

train = ds["train"]
print("Colonnes :", train.column_names)
print("Labels   :", train.features["label"].names)

df = train.to_pandas()
print("\nDistribution des classes :")
print(df["label"].map(train.features["label"].int2str).value_counts())

print("\nValeurs manquantes :\n", df.isna().sum())
print("Doublons (text) :", df["text"].duplicated().sum())
print("Longueur moyenne (mots) :", df["text"].str.split().str.len().mean())