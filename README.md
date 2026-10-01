# HarvestLink Cooperative: Harvest and Dispatch Decision Lab

SWE 3513 Artificial Intelligence, Assignment 1.
Institut d'Enseignement Supérieur de Ruhengeri (INES-Ruhengeri).

## 1. Group information

| Item                             | Value                                                              |
| -------------------------------- | ------------------------------------------------------------------ |
| Group number                     | G08                                                                |
| Group code (used with `--group`) | AI-G08                                                             |
| Group verification code          | AI-G08                                                             |
| Group leader                     | AISHA                                                              |
| GitHub repository                | https://github.com/lavi041/AI_A1_G08                               |
| Final project commit             | `2a232e4`                                                          |
| Dataset SHA-256                  | `5e62155fd4a2c78a98b3fd659c3a6826cae93e565994c6f9e689d270e8498fe7` |

### Members and roles

| Member | Name                        | Registration no. | Role                             |
| ------ | --------------------------- | ---------------- | -------------------------------- |
| 1      | Aisha Niyonsaba             | 25/27104         | Data and UX lead                 |
| 2      | Manzi Kassimu               | 25/27935         | Regression engineer              |
| 3      | Ineza Iwacu Adelphine       | 25/27676         | Classification engineer          |
| 4      | Ishimwe Sumaya              | 25/27949         | Clustering and QA engineer       |
| 5      | Mahgoub Adil Ahmed Alhassan | 25/28013         | Reproducibility and release lead |

## 2. Tested environment

* OS: Linux Ubuntu 64-bit
* Python: 3.14.4
* NumPy: 2.5.3
* pandas: 3.0.6
* scikit-learn: 1.9.1
* matplotlib: 3.11.2
* seaborn: 0.13.2

All external dependencies are listed in `requirements.txt`.

## 3. Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 4. Dataset

The project uses the group dataset:

```text
data/AI_A1_G08.csv
```

The dataset contains 100 synthetic HarvestLink records with the following fields:

* `record_id`
* `plot_area_ha`
* `rainfall_mm`
* `soil_ph`
* `seed_kg`
* `distance_km`
* `arrival_hour`
* `actual_yield_kg`
* `dispatch_attention`

The dataset can be reproduced using:

```bash
python create_dataset.py
```

Dataset SHA-256:

```text
5e62155fd4a2c78a98b3fd659c3a6826cae93e565994c6f9e689d270e8498fe7
```

## 5. Run the complete pipeline

```bash
python run_all.py --data data/AI_A1_G08.csv --output artifacts/ --group AI-G08
```

The pipeline performs:

1. Data loading and validation
2. Data preprocessing
3. Regression for harvest-yield prediction
4. Classification for dispatch attention
5. Clustering of records
6. Model evaluation
7. Artifact generation

## 6. Results

The completed pipeline produced:

### Regression

* Test MAE: `176.63`
* Test RMSE: `212.58`
* R²: `0.978`

### Classification

* Accuracy: `0.800`
* Precision: `0.833`
* Recall: `0.625`
* F1: `0.714`

### Clustering

Silhouette scores:

| k | Silhouette |
| - | ---------: |
| 2 |      0.223 |
| 3 |      0.196 |
| 4 |      0.188 |
| 5 |      0.175 |

The pipeline selected `k=2`.

## 7. Generated artifacts

The pipeline generates:

```text
artifacts/
├── classification_metrics.json
├── cluster_plot.png
├── clustering_metrics.json
├── clusters.csv
├── confusion_matrix.png
├── data_report.json
├── regression_loss.png
└── regression_metrics.json
```

## 8. Saved models

```text
models/
├── classification_model.joblib
├── clustering_model.joblib
├── model_info.json
└── regression_model.json
```

## 9. Prediction interface

A prediction can be requested using:

```bash
python predict.py --record '<JSON_RECORD>'
```

For command-line help:

```bash
python predict.py --help
```

## 10. Tests

The project includes automated tests in `tests/`.

Run:

```bash
PYTHONPATH=. pytest -q
```

Latest verification:

```text
17 passed in 6.62s
```

## 11. Project structure

```text
AI_A1_GXX/
├── artifacts/
├── data/
├── evidence/
├── models/
├── src/
├── tests/
├── .gitignore
├── create_dataset.py
├── predict.py
├── requirements.txt
├── run_all.py
└── README.md
```

## 12. Reproducibility

The dataset-generation script uses a fixed random seed so that the project dataset can be reproduced consistently.

The pipeline also records the dataset SHA-256 hash for integrity verification.

## 13. Evidence

Project evidence is stored in:

```text
evidence/
```

Current evidence includes:

```text
evidence/AI_USE.md
```

Additional evidence should be added according to the assignment submission requirements.

## 14. GitHub

Repository:

https://github.com/lavi041/AI_A1_G08

The main branch is synchronized with the remote repository.

Project commit containing the completed dataset, models, artifacts and core project implementation:

```text
2a232e4
```

The README finalization was subsequently committed as:

```text
dae8fb8
```

