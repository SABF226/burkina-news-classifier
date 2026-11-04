from datasets import load_dataset
import pandas as pd

ds = load_dataset("masakhane/masakhanews", "fra")
print(ds)                          # splits et tailles
print(ds["train"].features)        # colonnes + noms des labels

df = ds["train"].to_pandas()
names = ds["train"].features["label"].names
df["label_name"] = df["label"].map(lambda i: names[i])

print(df["label_name"].value_counts())            # distribution des classes
print(df.isna().sum())                            # valeurs manquantes
print("Doublons :", df["text"].duplicated().sum())
print(df["text"].str.split().str.len().describe())  # longueur en mots