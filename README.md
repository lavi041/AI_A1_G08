# Musanze HarvestLink Cooperative: Harvest and Dispatch Decision Lab

SWE 3513 Artificial Intelligence, Assignment 1.
Institut d'Enseignement Supérieur de Ruhengeri (INES-Ruhengeri).

## 1. Group information

| Item | Value |
|---|---|
| Group number | <<GXX>> |
| Group code (used with --group) | <<AI-GXX>> |
| Group verification code | <<CODE>> |
| Group leader | <<NAME>> |
| GitHub repository | <<URL>> |
| Final commit hash | <<HASH>> |
| Dataset SHA-256 | <<paste the value printed by run_all.py>> |

### Members and roles

| Member | Name | Registration no. | Role |
|---|---|---|---|
| 1 | <<NAME>> | <<REG>> | Data and UX lead |
| 2 | <<NAME>> | <<REG>> | Regression engineer |
| 3 | <<NAME>> | <<REG>> | Classification engineer |
| 4 | <<NAME>> | <<REG>> | Clustering and QA engineer |
| 5 | <<NAME>> | <<REG>> | Reproducibility and release lead (delete this row for a four-member group) |

## 2. Tested environment

- Python 3.14.4 on Linux (Ubuntu), 64-bit.
- numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2.
- All external dependencies are listed in `requirements.txt`.

## 3. Setup

From the project root:

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Run the full pipeline

```
python run_all.py --data data/AI_A1_GXX.csv --output artifacts/ --group AI-GXX
```

Replace GXX with the group number. The command prints the group code and the
dataset SHA-256 fingerprint, then runs the data, regression, classification and
clustering stages. Generated files are overwritten safely in `artifacts/` and
`models/`. The pipeline reads the row count, columns and values from the CSV;
nothing is hard-coded, so it also runs on another CSV with the same schema.

Optional settings (defaults in brackets):

| Option | Meaning |
|---|---|
| `--seed` [42] | Random seed for splits, KMeans and the classifier |
| `--test-size` [0.2] | Test fraction for regression and classification |
| `--lr` [0.1] | Learning rate for batch gradient descent |
| `--epochs` [2000] | Gradient descent iterations |
| `--threshold` [0.5] | Probability threshold for dispatch_attention = 1 |
| `--k-min`, `--k-max` [2, 5] | Range of k tested for clustering |
| `--models` [./models] | Folder for saved models |

## 5. Predict for one record

Run the pipeline first (it saves the models), then:

```
python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,"seed_kg":210,"distance_km":14,"arrival_hour":9}'
```

The output is JSON with `regression_prediction`, `classification_prediction`,
`classification_probability` (probability of class 1), `cluster_label`,
`group_code` and `model_version`.

A record is rejected with a clear JSON error on stderr (exit code 2) when it is
not valid JSON, is not an object, has a missing or unexpected field, or has a
value that is not a finite number.

## 6. Tests

```
python -m tests.test_regression
python -m tests.test_classification
python -m tests.test_clustering
python -m tests.test_predict
```

Each prints one PASS line per test.

## 7. Project structure

```
README.md  requirements.txt  run_all.py  predict.py
src/        data_pipeline.py  regression.py  classification.py
            clustering.py  model_info.py
tests/      test_regression.py  test_classification.py
            test_clustering.py  test_predict.py
data/       AI_A1_GXX.csv  (lecturer-issued, unchanged)
artifacts/  generated outputs
models/     saved models and preprocessing objects
evidence/   demo video, AI_USE.md, TEST_LOG.pdf
```

## 8. Expected outputs

`artifacts/`: `data_report.json`, `regression_metrics.json`, `regression_loss.png`,
`classification_metrics.json`, `confusion_matrix.png`, `clustering_metrics.json`,
`clusters.csv`, `cluster_plot.png`.

`models/`: `regression_model.json`, `classification_model.joblib`,
`clustering_model.joblib`, `model_info.json`.

## 9. Method summary

**Data.** Column names and types are validated against the published schema.
Missing values and duplicates are counted in `data_report.json`. Exact duplicate
rows and rows with a missing value are then removed. `record_id` is never a
feature. The feature matrix has six columns: plot_area_ha, rainfall_mm, soil_ph,
seed_kg, distance_km, arrival_hour.

**Regression (NumPy only).** Linear regression trained with batch gradient
descent. Loss: J(w) = 1/(2n) * sum((y_hat - y)^2). Features and target are
standardized with training-set statistics only. The loss history, MAE, RMSE and
R-squared are saved.

**Classification (scikit-learn).** Logistic regression in a pipeline with a
StandardScaler fitted on the training split only. Stratified split when
possible. Confusion matrix, accuracy, precision, recall and F1 are reported on
the test set. The error-cost discussion is in `classification_metrics.json`.

**Clustering.** KMeans on the standardized input features only (targets and
record_id excluded). k from 2 to 5 is compared by silhouette score and the best
k is selected. Every record receives a label in `clusters.csv`. Clusters are
statistical groups, not verified real-world categories.

## 10. Known limitations

- Rows with missing values are dropped, not imputed, and exact duplicates are removed.
- The CSV must contain exactly the published columns; extra columns are rejected.
- `predict.py` checks field presence, type and finiteness, but not domain ranges
  (for example, it does not reject arrival_hour = 30).
- Regression is purely linear and evaluated on a single hold-out split; there is
  no cross-validation and the learning rate and epochs are fixed defaults.
- Classification uses a fixed 0.5 threshold, no class re-weighting, and a single
  split, so metrics on small test sets can be noisy.
- Clustering quality depends on the data; when the best silhouette score is below
  0.25 the metrics file flags the structure as weak. The scaler is fitted on all
  rows because clustering is unsupervised.
- The models are trained on a fictional dataset and are not suitable for real
  operational decisions.
- Tested only on Python 3.14.4 under Linux.
