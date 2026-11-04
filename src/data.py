"""Load and prepare the MasakhaNEWS French dataset splits for Burkina News Classifier.

Responsible: Team Member 1 (Dataset Research & Data Preparation)
"""

import pandas as pd
import numpy as np
import logging
import hashlib
from pathlib import Path
from typing import Tuple, List

logger = logging.getLogger(__name__)

# Fix random seeds for reproducibility
SEED = 42
np.random.seed(SEED)


def _text_hash(text: str) -> str:
    """Compute a hash for a text to detect duplicates across splits."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _log_counts(prefix: str, dfs: dict):
    """Log the number of rows in each split."""
    for split_name, df in dfs.items():
        logger.info(f"{prefix} {split_name}: {len(df)} rows")


def load_splits(data_dir: str = "datasets", seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation and test splits with cleaning rules.

    Returns:
        Tuple of (train_df, val_df, test_df) with columns:
        - text: article text
        - label: numeric label index (0-4)
        - label_name: string label name

    Cleaning rules applied (in order, first wins):
    1. Drop exact duplicate rows (all columns identical) within train; keep first occurrence.
    2. For every text appearing >1 occurrence:
         a. If labels differ across occurrences → remove from ALL splits (train + others).
            This catches cross-split leakage AND intra-train conflicts.
         b. If labels are SAME across all occurrences → remove only from val/test, keep in train.
    3. Final verification: no text overlap between any two splits; 5 classes everywhere.
    4. Log line counts before and after cleaning.
    """

    # ── Load raw CSVs ──
    train_df = pd.read_csv(f"{data_dir}/masakhanews_fr_train.csv", dtype=str)
    val_df = pd.read_csv(f"{data_dir}/masakhanews_fr_validation.csv", dtype=str)
    test_df = pd.read_csv(f"{data_dir}/masakhanews_fr_test.csv", dtype=str)

    # Ensure column names are as expected
    expected_cols = ["label_name", "headline", "text", "url"]
    for df_to_rename, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        if list(df_to_rename.columns) != expected_cols:
            raise ValueError(f"{name} columns mismatch: expected {expected_cols}, got {list(df_to_rename.columns)}")

    original_counts = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    _log_counts("Before cleaning -> ", {"train": train_df, "val": val_df, "test": test_df})

    # ── Rule 1: Remove exact duplicate rows within train (keep first) ──
    train_dups = train_df.duplicated(keep="first").sum()
    train_df = train_df[~train_df.duplicated(keep="first")].reset_index(drop=True)
    logger.info(f"Removed {train_dups} exact duplicate rows from train")

    # ── Rule 2 & 3: Find shared texts across splits ──
    # Compute hashes for each text across all splits
    all_texts_info = {}  # hash -> [(split, index, label_name)]

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for idx, row in df.iterrows():
            text_h = _text_hash(row["text"])
            if text_h not in all_texts_info:
                all_texts_info[text_h] = []
            all_texts_info[text_h].append((split_name, idx, row["label_name"]))

    # Identify shared texts and their label situations
    shared_across_splits = []  # list of (split, index, label_name, should_remove)
    for text_h, occurrences in all_texts_info.items():
        if len(occurrences) < 2:
            continue

        splits_present = {o[0] for o in occurrences}
        labels_present = {o[2] for o in occurrences}

        # Case A: Same label in all occurrences -> remove from val/test only, keep in train
        if len(labels_present) == 1:
            # Remove from val and/or test
            for split_name, idx, label in occurrences:
                if split_name in ("val", "test"):
                    shared_across_splits.append((split_name, idx, label, True))

        # Case B: Different labels across occurrences -> remove from ALL splits
        else:
            for split_name, idx, label in occurrences:
                shared_across_splits.append((split_name, idx, label, True))

    # Apply removals
    removals_by_split = {"train": set(), "val": set(), "test": set()}
    for split_name, idx, label, should_remove in shared_across_splits:
        if should_remove:
            removals_by_split[split_name].add(idx)

    # Filter out removed rows
    train_df = train_df[~train_df.index.isin(removals_by_split["train"])].reset_index(drop=True)
    val_df = val_df[~val_df.index.isin(removals_by_split["val"])].reset_index(drop=True)
    test_df = test_df[~test_df.index.isin(removals_by_split["test"])].reset_index(drop=True)

    # Log removals
    for split_name in ["train", "val", "test"]:
        n_removed = len(removals_by_split[split_name])
        if n_removed > 0:
            logger.info(f"Removed {n_removed} shared-text rows from {split_name}")

    final_counts = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    logger.info(f"After cleaning -> train: {final_counts['train']}, val: {final_counts['val']}, test: {final_counts['test']}")

    # ── Verify no text overlap between splits ──
    train_texts = set(train_df["text"].astype(str))
    val_texts = set(val_df["text"].astype(str))
    test_texts = set(test_df["text"].astype(str))

    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts

    if len(train_val_overlap) > 0:
        logger.warning(f"{len(train_val_overlap)} text(s) overlap between train and val after cleaning")
    if len(train_test_overlap) > 0:
        logger.warning(f"{len(train_test_overlap)} text(s) overlap between train and test after cleaning")
    if len(val_test_overlap) > 0:
        logger.warning(f"{len(val_test_overlap)} text(s) overlap between val and test after cleaning")

    # ── Final verification ──
    if len(train_df["label_name"].unique()) != 5:
        raise ValueError(f"Expected 5 classes in train, found {len(train_df['label_name'].unique())}")
    if len(val_df["label_name"].unique()) != 5:
        raise ValueError(f"Expected 5 classes in val, found {len(val_df['label_name'].unique())}")
    if len(test_df["label_name"].unique()) != 5:
        raise ValueError(f"Expected 5 classes in test, found {len(test_df['label_name'].unique())}")

    if train_df["text"].isna().any() or val_df["text"].isna().any() or test_df["text"].isna().any():
        raise ValueError("Found missing values in text column after cleaning")

    _log_counts("Final -> ", {"train": train_df, "val": val_df, "test": test_df})

    # ── Log summary with detailed removals ──
    logger.info("=" * 50)
    logger.info("DATA LOAD SUMMARY — REMOVAL BREAKDOWN")
    logger.info("=" * 50)
    logger.info(f"Original counts  -> train: {original_counts['train']}, val: {original_counts['val']}, test: {original_counts['test']}")
    logger.info(f"Final counts     -> train: {final_counts['train']}, val: {final_counts['val']}, test: {final_counts['test']}")
    logger.info(f"Rule 1 (internal dups) removed from train: {train_dups}")
    logger.info(f"Rule 2 (diff labels, all splits) removed -> train: {len(removals_by_split['train'])}, val: {len(removals_by_split['val'])}, test: {len(removals_by_split['test'])}")
    logger.info("=" * 50)

    # Rename label_name -> label (numeric) and keep label_name for reference
    label_map = {"business": 0, "health": 1, "politics": 2, "sports": 3, "technology": 4}
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["label"] = train_df["label_name"].map(label_map)
    val_df["label"] = val_df["label_name"].map(label_map)
    test_df["label"] = test_df["label_name"].map(label_map)

    # Ensure no NaN after mapping
    if train_df["label"].isna().any() or val_df["label"].isna().any() or test_df["label"].isna().any():
        raise ValueError("Some label_name values could not be mapped to numeric labels")

    # Return only text, label, label_name
    train_df = train_df[["text", "label", "label_name"]]
    val_df = val_df[["text", "label", "label_name"]]
    test_df = test_df[["text", "label", "label_name"]]

    return train_df, val_df, test_df


if __name__ == "__main__":
    """Allow running as script: python -m src.data"""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train, val, test = load_splits()
    print(f"Train: {len(train)} rows | Classes: {sorted(train['label_name'].unique())}")
    print(f"Val:   {len(val)} rows | Classes: {sorted(val['label_name'].unique())}")
    print(f"Test:  {len(test)} rows | Classes: {sorted(test['label_name'].unique())}")
    print(f"\nTrain labels distribution:")
    print(train["label_name"].value_counts().sort_index())
    print(f"\nVal labels distribution:")
    print(val["label_name"].value_counts().sort_index())
    print(f"\nTest labels distribution:")
    print(test["label_name"].value_counts().sort_index())