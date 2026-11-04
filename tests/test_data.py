"""Tests for src/data.py — data loading and cleaning module.

Responsible: Team Member 1 (Dataset Research & Data Preparation)
"""

import sys
from pathlib import Path

# Ensure src/ is importable when running pytest from project root
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data import load_splits


def test_load_splits_returns_three_dataframes():
    """load_splits returns exactly three pandas DataFrames."""
    train, val, test = load_splits()
    assert isinstance(train, object), "train is not a DataFrame"
    assert isinstance(val, object), "val is not a DataFrame"
    assert isinstance(test, object), "test is not a DataFrame"


def test_load_splits_has_correct_columns():
    """Each DataFrame has columns text, label, label_name."""
    train, val, test = load_splits()
    expected_cols = {"text", "label", "label_name"}
    for df_name, df in [("train", train), ("val", val), ("test", test)]:
        assert set(df.columns) == expected_cols, f"{df_name} columns: {set(df.columns)}"


def test_five_classes_present_in_each_split():
    """All 5 classes (business, health, politics, sports, technology) appear in each split."""
    expected_classes = {"business", "health", "politics", "sports", "technology"}
    train, val, test = load_splits()
    for df_name, df in [("train", train), ("val", val), ("test", test)]:
        assert set(df["label_name"].unique()) == expected_classes, (
            f"{df_name} classes: {set(df['label_name'].unique())}"
        )


def test_no_shared_text_between_splits():
    """No article text appears in more than one split after cleaning."""
    train, val, test = load_splits()

    train_texts = set(train["text"].astype(str))
    val_texts = set(val["text"].astype(str))
    test_texts = set(test["text"].astype(str))

    assert len(train_texts & val_texts) == 0, f"train/val overlap: {len(train_texts & val_texts)}"
    assert len(train_texts & test_texts) == 0, f"train/test overlap: {len(train_texts & test_texts)}"
    assert len(val_texts & test_texts) == 0, f"val/test overlap: {len(val_texts & test_texts)}"


def test_no_missing_values():
    """No missing values in text, label, or label_name columns."""
    train, val, test = load_splits()
    for df_name, df in [("train", train), ("val", val), ("test", test)]:
        assert df["text"].notna().all(), f"{df_name}: missing values in text"
        assert df["label"].notna().all(), f"{df_name}: missing values in label"
        assert df["label_name"].notna().all(), f"{df_name}: missing values in label_name"
        assert (df["text"].str.strip() != "").all(), f"{df_name}: empty text after strip"


def test_label_mapping_consistent():
    """label values match label_name consistently."""
    train, val, test = load_splits()
    expected = {"business": 0, "health": 1, "politics": 2, "sports": 3, "technology": 4}
    for df_name, df in [("train", train), ("val", val), ("test", test)]:
        for _, row in df.iterrows():
            assert row["label"] == expected[row["label_name"]], (
                f"{df_name}: label {row['label']} != {expected[row['label_name']]} for {row['label_name']}"
            )