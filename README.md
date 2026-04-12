# NYC Restaurant Health Inspection — Score Prediction

A machine learning project to predict restaurant health inspection scores in New York City, using structured violation data, geographic features, and natural language processing on violation descriptions — with rigorous data leakage prevention throughout the pipeline.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)
![License](https://img.shields.io/badge/License-MIT-green) 

---

## Objective

The NYC Department of Health assigns each restaurant a numeric score during inspections — higher scores mean more or worse violations. The goal of this project is to **predict that score** from features available at inspection time: the types and severity of violations recorded, the restaurant's location, the inspection type, and the text of violation descriptions.

A reliable predictor can help:
- Flag high-risk restaurants before re-inspection
- Understand which violation patterns drive scores up
- Identify boroughs or cuisine types with elevated risk profiles

---

## Dataset

**Source:** [NYC DOHMH Restaurant Inspection Results](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j) — NYC Open Data

Each row in the raw data represents **one violation** within one inspection visit. A single inspection can produce multiple rows.

| Property | Value |
|---|---|
| Raw records | 285,749 rows |
| Raw columns | 27 |
| Date range | 2015 – 2025 |
| After aggregation (one row per inspection visit) | 80,006 rows |
| Training set | 64,004 rows (80%) |
| Test set | 16,002 rows (20%) |
| Target variable | `SCORE` — continuous, range 0–153 |

### Key Columns

| Column | Type | Description |
|---|---|---|
| `CAMIS` | int | Unique restaurant identifier |
| `BORO` | categorical | Borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island) |
| `CUISINE DESCRIPTION` | categorical | Type of cuisine |
| `INSPECTION DATE` | date | Date of the inspection visit |
| `INSPECTION TYPE` | categorical | Initial / Re-inspection / Compliance / etc. |
| `VIOLATION CODE` | categorical | Code of the specific violation cited |
| `VIOLATION DESCRIPTION` | text | Free-text description of the violation |
| `CRITICAL FLAG` | categorical | Whether the violation is critical |
| `SCORE` | float | **Target** — total penalty points for the visit |
| `GRADE` | categorical | Letter grade derived from score (A / B / C) |
| `Latitude`, `Longitude` | float | Geographic coordinates |

---

## Data Cleaning

| Step | Detail |
|---|---|
| Dropped columns | `Location Point1` (entirely empty), `BIN`, `BBL`, `GRADE DATE` |
| `GRADE DATE` | Verified 100% match with `INSPECTION DATE` across 129,941 rows — zero mismatches — safely dropped |
| Missing `SCORE` | Dropped — these rows belong to administrative inspection types that produce no score |
| Missing `GRADE` | Reconstructed from `SCORE` using official NYC grading rules (see below) |
| Missing `ZIPCODE` | Imputed with median |
| Missing `PHONE` | Imputed with mode |
| Missing violation fields | Filled with `'NoViolation'` |
| Missing coordinates | Rows dropped |
| **Final missing values** | **0 across all columns** |

**NYC Grading Rules used for reconstruction:**

| Score Range | Grade |
|---|---|
| 0 | P (Pass) |
| 1 – 13 | A |
| 14 – 27 | B |
| ≥ 28 | C |

---

## Exploratory Data Analysis

### Score Distribution
- Right-skewed: most restaurants score between 0 and 40
- Most common scores: 12 and 13 (just under the A/B boundary at 13)
- Median around 10–13, indicating most restaurants perform well
- A small tail of extreme cases exceeds 60 or even 100

### Grade Distribution
- Majority of restaurants receive grade **A** (score ≤ 13)
- B and C grades represent a smaller but significant share
- Grades N (not yet graded) and Z (no grade) appear for certain inspection types

### Score by Borough
- All five boroughs show broadly similar score distributions
- Manhattan and Brooklyn show slightly higher variance
- Bronx and Queens tend toward slightly lower scores on average

### Score Over Time
- Clear **COVID-19 gap** visible in mid-2020: inspections were largely suspended
- Post-2021: a visible **upward trend** in average scores — reflecting either stricter enforcement or post-pandemic compliance issues
- Monthly averages stabilize from 2022 onward

### Critical Violations
- The majority of inspections include at least one critical violation
- Chinese, Latin American, and Mexican cuisines show higher critical violation rates among the top-10 cuisine types
- All five boroughs share similarly elevated critical violation proportions — this is a city-wide pattern, not borough-specific

### Violation Codes
- A small set of violation codes accounts for the vast majority of citations
- Rare codes (frequency < 1%) were grouped into an `'Other'` category before feature construction

### Correlation Analysis (Cramér's V)
- `violation_count` and `critical_count` show the strongest correlation with `SCORE`
- Geographic and administrative columns show near-zero linear correlation with score — but non-linear models still exploit them
- Cramér's V between `INSPECTION TYPE` and `CRITICAL FLAG`: **0.445** (strong association)
- Cramér's V between `GRADE` and `INSPECTION TYPE`: **0.249** (moderate)

---

## Feature Engineering

The raw data has one row per violation. All features were computed after aggregating to **one row per inspection visit** (`CAMIS` + `INSPECTION DATE`).

### Violation Features

| Feature | Description |
|---|---|
| `violation_count` | Total violations recorded in the visit |
| `critical_count` | Number of critical violations |
| `FLAG_CRITICAL` | Binary: 1 if any critical violation exists |
| `has_critical` | Binary: 1 if `critical_count > 0` |
| `viol_ratio` | `critical_count / (violation_count + 1)` |
| `viol_level` | Severity bucket: Low (≤2) / Medium (3–5) / High (6–10) / Very High (>10) |
| `days_since_last` | Days elapsed since the restaurant's previous inspection visit |

### Temporal Features

Extracted from `INSPECTION DATE`:

`year`, `month`, `day`, `dayofweek`, `is_weekend`, `is_summer`, `season`

### Geographic Features

| Feature | Description |
|---|---|
| `ZIPCODE` | Numeric postal code |
| `ZIP3` | First 3 digits of ZIP code — regional grouping |
| `geo_cluster` | KMeans (k=20) on `(Latitude, Longitude)` — captures sub-borough local patterns |

### Text Features (Violation Descriptions)

Violation descriptions for each visit were concatenated into one document per inspection, then processed through a two-step NLP pipeline:

1. **TF-IDF** — `TfidfVectorizer(min_df=0.01, max_df=0.90, ngram_range=(1,2))` extracts unigram and bigram features, weighting rare but informative terms higher
2. **TruncatedSVD** — reduces the sparse TF-IDF matrix to **50 latent semantic components**, avoiding the memory explosion that `sklearn.PCA` causes on dense matrices

All text transformations are fitted on training data only, inside the Pipeline.

---

## Modeling

### Pipeline Architecture

All preprocessing is encapsulated in a single `sklearn.Pipeline` to prevent any data leakage:

```
X_train / X_test
     │
     ▼
CombinedFeatures (custom transformer)
  ├── ColumnTransformer
  │     ├── StandardScaler        →  numeric columns
  │     ├── OneHotEncoder         →  categorical columns
  │     └── GeoCluster (custom)   →  (Latitude, Longitude) → geo_cluster
  └── Text Pipeline
        ├── ColumnSelector        →  'joined_description'
        ├── TfidfVectorizer       →  sparse term matrix
        └── TruncatedSVD          →  50 dense components
     │
     ▼
  Model  (LinearRegression / CatBoost / XGBoost / ...)
```

`GeoCluster` and `TfidfVectorizer` learn their parameters only from `X_train`. `X_test` is only ever transformed, never used to fit.

### Results

| Model | Train R² | Test R² | RMSE | MAE |
|---|---|---|---|---|
| **CatBoost Regressor** | 0.864 | **0.835** | **5.939** | **2.820** |
| XGBoost Regressor | 0.878 | 0.833 | 5.972 | 2.815 |
| LightGBM Regressor | 0.859 | 0.830 | 6.022 | 2.868 |
| Random Forest | 0.833 | 0.806 | 6.441 | 3.118 |
| Ridge Regression | 0.800 | 0.800 | 6.539 | 3.258 |
| Linear Regression | 0.800 | 0.800 | 6.539 | 3.261 |
| Decision Tree | 0.758 | 0.757 | 7.198 | 3.785 |
| KNN Regressor | 0.735 | 0.706 | 7.928 | 4.333 |

### Best Model — CatBoost + Optuna

Hyperparameters tuned with **Optuna** (Bayesian optimization, 10 trials). Each trial evaluated using `cross_val_score(cv=3)` on `X_train` — the test set was never touched during the search.

**Best hyperparameters found:**

| Parameter | Value |
|---|---|
| `iterations` | 600 |
| `learning_rate` | 0.159 |
| `depth` | 5 |
| `l2_leaf_reg` | 3.10 |

**Final evaluation on held-out test set:**

| Metric | Value | Interpretation |
|---|---|---|
| Train R² | 0.892 | Moderate overfitting — acceptable gap |
| **Test R²** | **0.832** | Model explains 83.2% of score variance |
| **Test RMSE** | **5.993** | Average error ≈ 6 inspection points |
| **Test MAE** | **2.840** | Median error ≈ 3 inspection points |

### 5-Fold Cross-Validation

| Fold | R² | RMSE | MAE |
|---|---|---|---|
| Fold 1 | 0.804 | 5.416 | 2.506 |
| Fold 2 | 0.790 | 6.035 | 2.769 |
| Fold 3 | 0.785 | 6.416 | 3.017 |
| Fold 4 | 0.824 | 6.120 | 2.906 |
| Fold 5 | 0.876 | 6.131 | 3.028 |
| **Mean** | **0.816 ± 0.033** | **6.024** | **2.845** |

The low standard deviation (±0.033) confirms stable generalization across folds.

---

## Data Leakage — Issues Found and Fixed

An earlier version of this project had leakage in four places. All were corrected before final evaluation:

| # | Issue | Root Cause | Fix Applied |
|---|---|---|---|
| 1 | TF-IDF fitted on all data | `vec.fit_transform(agg)` ran before `train_test_split` | Moved inside `Pipeline` — fitted on `X_train` only |
| 2 | KMeans fitted on all data | `kmeans.fit_predict(agg)` ran before split | Custom `GeoCluster` transformer inside `Pipeline` |
| 3 | Optuna evaluating on random splits | Each trial called `train_test_split(agg)` independently | Replaced with `cross_val_score(cv=3, X_train)` per trial |
| 4 | PCA on dense TF-IDF matrix | `sklearn.PCA` required `.toarray()` → ~4 GB RAM | Replaced with `TruncatedSVD` — works on sparse directly |

Without these fixes, the inflated R² from the original code (~0.85+) was not trustworthy. The corrected pipeline yields an honest **Test R² = 0.832**.

---

## Key Findings

- `violation_count` and `critical_count` are by far the strongest predictors of inspection score
- Text features (TF-IDF + SVD on violation descriptions) meaningfully improve tree-based model performance beyond structured features alone
- Geographic clusters capture neighborhood-level patterns that ZIP code alone misses
- Inspection scores trended upward post-2021 — combining stricter scoring with post-COVID compliance issues
- All tree-based ensemble models (CatBoost, XGBoost, LightGBM) comfortably outperform linear baselines, confirming non-linear relationships in the data
- The Train/Test R² gap (~0.06) is consistent across models — no severe overfitting

---

## Technologies

| Category | Tools |
|---|---|
| Environment | Google Colab |
| Data processing | `pandas`, `numpy` |
| Machine learning | `scikit-learn`, `xgboost`, `lightgbm`, `catboost` |
| NLP / text features | `TfidfVectorizer`, `TruncatedSVD` (scikit-learn) |
| Hyperparameter tuning | `optuna` |
| Visualization | `matplotlib`, `seaborn` |


---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).  
The dataset is provided by NYC Open Data under the [NYC Open Data Terms of Use](https://opendata.cityofnewyork.us/overview/#termsofuse).
