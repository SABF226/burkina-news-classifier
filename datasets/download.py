from datasets import load_dataset

ds = load_dataset("masakhane/masakhanews", "fra")
print(ds)

for split in ds:
    df = ds[split].to_pandas()
    df = df.rename(columns={"category": "label_name"})
    df.to_csv(
        f"masakhanews_fr_{split}.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(split, len(df))
    print(df["label_name"].value_counts())