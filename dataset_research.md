# Dataset Research Audit — Burkina News Classifier

## 1. Dataset Overview
- **Dataset name**: `masakhanews_fr` (Francophone version of MasakhaNEWS)
- **Source**: Hugging Face `masakhane/masakhanews` (dataset_id: `masakhane/masakhanews`, config: `fra`)
- **License**: Not specified in dataset metadata (empty field)
- **Provenance**: Files downloaded via script `download.py` using `datasets.load_dataset("masakhane/masakhanews", "fra")`, then saved as CSV files (`masakhanews_fr_train.csv`, `masakhanews_fr_validation.csv`, `masakhanews_fr_test.csv`).

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
- **Exact duplicates (all columns) in train**: 2 rows (indices 914 and 1427) - identical content across all columns
- **Shared texts between splits**:
  - train ∩ test: 1 article (health in train → business in test, index 209 in train, 53 in test)
  - validation ∩ test: 1 article (business in validation → health in test, index 129 in validation, 156 in test)
  - train ∩ validation: 0 shared texts

### 4.4 Text Length Statistics (words)
#### Train split
- min: 12
- median: 717.5
- mean: 818.4
- 90th percentile: 1597.0
- 95th percentile: 1948.8
- max: 7807

#### Validation split
- min: 1
- median: 662.0
- mean: 766.9
- 90th percentile: 1498.0
- 95th percentile: 2021.5
- max: 3474

#### Test split
- min: 35
- median: 674.5
- mean: 827.6
- 90th percentile: 1676.1
- 95th percentile: 2156.7
- max: 4092

## 5. Dataset Limitations
1. **Class imbalance**: The `technology` class is under-represented (76 train, 11 val, 22 test) compared to other classes (350 train, 50 val, 100 test).
2. **Class mismatch**: The dataset contains 5 classes, but the project requires 5 classes (business, health, politics, sports, technology). All classes are relevant except that `technology` should be excluded from training/validation/test splits as it's not part of the 4 target categories.
3. **Data leakage**: One article appears in both train and test splits (different labels), indicating potential leakage. This must be resolved by removing the overlapping row(s) before training.
4. **License**: Not specified in dataset metadata (empty field). The Masakhane organization typically uses permissive licenses, but verification is needed before public release.

## 6. Recommended Next Actions
1. **Exclude `technology` class** from all splits (train, validation, test) to align with project requirements.
2. **Remove duplicate rows** between train and test (1 article) and between validation and test (1 article) to prevent data leakage.
3. **Re-split the dataset** after exclusions to maintain proper class distribution across the 5 classes.
4. **Document the final class mapping** and ensure all preprocessing steps respect the class structure.
5. **Verify license** on the Hugging Face dataset page to confirm usage permissions for educational purposes.

---

*Document generated on Wed Nov 04 2026. All data extracted via executed Python commands and direct inspection of CSV files.*