# Dataset Research Audit — Burkina News Classifier

## 1. Dataset Overview
- **Dataset name**: `masakhanews_fr` (Francophone version of MasakhaNEWS)
- **Source**: Hugging Face `masakhane/masakhanews` (dataset_id: `masakhane/masakhanews`, config: `fra`)
- **License**: Not specified in dataset metadata (empty field). The Masakhane organization typically uses permissive licenses (MIT) but this must be verified on the Hugging Face dataset page before public release.
- **Provenance**: Files downloaded via script `download.py` using `datasets.load_dataset("masakhane/masakhanews", "fra")`, then saved as CSV files (`masakhanews_fr_train.csv`, `masakhanews_fr_validation.csv`, `masakhanews_fr_test.csv`).
- **Limit stated**: Articles are in French but not specifically Burkina Faso news. They cover multiple African countries.

## 2. Dataset Structure
### 2.1 Columns
| Column | Description |
|--------|-------------|
| `label_name` | Target class label (string) |
| `headline` | Article headline (string) |
| `text` | Full article text (string) |
| `url` | URL of the article (string) |

### 2.2 Label Mapping (index → name)
- `0` → `business`
- `1` → `health`
- `2` → `politics`
- `3` → `sports`
- `4` → `technology`

## 3. Dataset Splits and Distribution
### 3.1 Split Sizes
| Split | Number of rows | Classes |
|-------|----------------|---------|
| **train** | 1,476 | business(350), health(350), politics(350), sports(350), technology(76) |
| **validation** | 211 | business(50), health(50), politics(50), sports(50), technology(11) |
| **test** | 422 | business(100), health(100), politics(100), sports(100), technology(22) |

### 3.2 Class Distribution (value_counts)
#### Train split
```
label_name
sports        350
business      350
politics      350
health        350
technology     76
Name: count, dtype: int64
```

#### Validation split
```
label_name
business      50
sports        50
politics      50
health        50
technology    11
Name: count, dtype: int64
```

#### Test split
```
label_name
business      100
health        100
politics      100
sports        100
technology     22
Name: count, dtype: int64
```

## 4. Data Quality Checks
### 4.1 Missing Values
| Split | Missing values (per column) |
|-------|-----------------------------|
| train | 0 (all columns) |
| validation | 0 (all columns) |
| test | 0 (all columns) |

### 4.2 Empty Texts
- No empty `text` or `headline` values after stripping whitespace (`""` or whitespace-only) in any split.

### 4.3 Duplicates
- **Exact duplicates (all columns) in train**: 1 row to drop (index 1427 is an exact duplicate of index 914, same label=business, same headline, same url, same text).
- **Shared texts between splits**:
  - train ∩ test: 1 article (labels differ: health vs business — both share same text). Removed from ALL splits. Indices: train[209], test[53].
  - validation ∩ test: 1 article (labels differ: business vs health — both share same text). Removed from val+test. Indices: val[129], test[156].
  - train ∩ validation: 0 shared texts.
- **Intra-train conflicts**: 2 articles with identical text but DIFFERENT labels (both labels reference the same body text, e.g. row 933 labeled business shares text with row 1096 labeled health). Removed from train only. Indices: train[933], train[1096].

### 4.4 Text Length Statistics (words) — after cleaning
#### Train split (1,472 rows after deduplication + conflict removal)
- min: 12
- median: 717.5
- mean: 818.4
- 90th percentile: 1597.0
- 95th percentile: 1948.8
- max: 7807

#### Validation split (210 rows)
- min: 1
- median: 662.0
- mean: 766.9
- 90th percentile: 1498.0
- 95th percentile: 2021.5
- max: 3474

#### Test split (420 rows)
- min: 35
- median: 674.5
- mean: 827.6
- 90th percentile: 1676.1
- 95th percentile: 2156.7
- max: 4092

## 5. Dataset Limitations
1. **Class imbalance**: The `technology` class is under-represented (76 train, 11 val, 22 test) compared to other classes (350 train, 50 val, 100 test). This will affect model performance on this class.
2. **Data leakage**: One article appears in both train and test splits with DIFFERENT labels (leaks a wrong label into test evaluation). Resolved by removal before training. Same for val∩test. Intra-train conflicts also resolved.
3. **Language**: Articles are in French but not specifically Burkina Faso news. Further filtering may be needed for a pure burkinabè focus.
4. **License**: Not specified in dataset metadata (empty field). Must verify on Hugging Face page before public release.

## 6. Recommended Next Actions
1. **Keep all 5 classes** as project requires (no deletion requested).
2. **Clean splits** using the documented rules (exact dups, shared-text conflicts, intra-train conflicts) — already implemented in `src/data.py`.
3. **Verify license** on the Hugging Face dataset page.
4. Proceed to Phase 2 of implementation: tokenization and vocabulary building.

---

*Document generated on Wed Nov 04 2026. All data extracted via executed Python commands and direct inspection of CSV files.*