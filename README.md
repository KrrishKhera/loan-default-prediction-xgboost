# Loan Default Prediction Using XGBoost

A reproducible machine learning application for estimating borrower default risk after loan terms have been set and before repayment begins. It includes model evaluation, SHAP explanations, a FastAPI prediction service, and a modular Streamlit dashboard.

## Current status

The project includes reproducible model comparison, preprocessing, the saved XGBoost pipeline, SHAP explanations, a FastAPI service, and a Streamlit dashboard with Overview, Risk assessment, Model analytics, and Explainability sections. Logistic Regression, Random Forest, and XGBoost use training-only cross-validation and validation-based model and threshold selection. The old test partition was evaluated multiple times; its metrics are **preliminary development diagnostics**, not pristine final holdout results. See [docs/training.md](docs/training.md) and [docs/model_results.md](docs/model_results.md).

Measured results from the current run are summarized in [docs/model_results.md](docs/model_results.md); CSV/JSON details are saved under `reports/`.

Exploratory SHAP explanations for the saved model are described in [docs/explainability.md](docs/explainability.md) and saved under `reports/exploratory_shap/`.

The Stage 5 FastAPI service uses the saved model and SHAP module. Run it locally with `PYTHONPATH=src uvicorn api.app:app --reload`; interactive docs are at `http://127.0.0.1:8000/docs`. The explanation route uses an 80-row deterministic background sampled from the development training partition in `data/raw/credit_risk_dataset.csv`. See the API run instructions below.

## Dataset

Download the **Credit Risk Dataset** from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) and place `credit_risk_dataset.csv` at `data/raw/credit_risk_dataset.csv`. The raw file is excluded from Git. Do not put Kaggle credentials in this repository.

The expected target is `loan_status` (`0` = non-default, `1` = default). The loader checks the required schema and valid target values; feature missing values are retained for the preprocessing pipeline to impute.

### Fresh-clone data and model requirements

The raw CSV and fitted model are intentionally excluded from Git. Download the CSV from the Kaggle page above and place it at `data/raw/credit_risk_dataset.csv`. The tracked aggregate reports and global SHAP summary support the analytics and explainability views, but dataset-derived Overview statistics and API predictions require the local CSV and model. The API can start without them, but `/health` reports unavailable readiness and `/predict` returns an unavailable response until a compatible model is configured. Individual `/explain` requests also need the CSV as background data.

To create a local model artifact from the downloaded CSV, use a separate ignored output directory:

```bash
mkdir -p local_artifacts
PYTHONPATH=src .venv/bin/python -m loan_default.training \
  --csv data/raw/credit_risk_dataset.csv \
  --output-dir local_artifacts \
  --random-state 42 \
  --cv-splits 3
```

This command **re-trains** the candidate models and writes new local model artifacts and reports under `local_artifacts/`; it does not reproduce or restore the previously evaluated model or its historical results. It also evaluates the previously viewed test partition, so those outputs remain development diagnostics and must not be described as an untouched final holdout. The command is documented for local setup only and has not been run as part of this repository audit.

## Setup

Use the project-local virtual environment so these dependencies stay separate from other Python projects. The current saved model was trained with Python 3.13.1; use Python 3.13 for the closest runtime match.

```bash
python3 -m venv .venv
source .venv/bin/activate
export PIP_CACHE_DIR="$PWD/.venv/.cache/pip"
export MPLCONFIGDIR="$PWD/.venv/.cache/matplotlib"
.venv/bin/python -m pip install --cache-dir .venv/.cache/pip -r requirements.txt
```

The virtual environment is ignored by Git. Verify the active project interpreter and package location with:

```bash
.venv/bin/python -c 'import sys; print(sys.executable); print("isolated:", sys.prefix != sys.base_prefix)'
.venv/bin/python -m pip --version
.venv/bin/python -m pip show numpy pandas scikit-learn xgboost shap fastapi streamlit
```

Do not install project dependencies with a shared or system Python. Pip's download cache is also directed into `.venv/.cache/pip`. On macOS, XGBoost uses the OpenMP runtime; the runtime is a system library, separate from Python packages, and should only be installed through your normal Homebrew process if XGBoost reports that it is missing.

### VS Code interpreter

Open this project directory, run **Python: Select Interpreter** from the Command Palette, and select `.venv/bin/python` (the full path is `<project>/.venv/bin/python`). The workspace setting in `.vscode/settings.json` points VS Code to this interpreter and asks its terminals to activate it. For notebook files, select the same `.venv` interpreter as the notebook kernel.

## Validate and split the dataset

```bash
.venv/bin/python -m loan_default.data --csv data/raw/credit_risk_dataset.csv
```

For development without installing the package, set the source path:

```bash
PYTHONPATH=src .venv/bin/python -m loan_default.data --csv data/raw/credit_risk_dataset.csv
```

The split helper in `loan_default.data` returns stratified train, validation, and test partitions. Rows with identical predictors are kept together in one partition and CV fold to reduce contamination; no records are deleted. Use `random_state` to reproduce a split.

## Train and evaluate

No genuinely untouched final holdout currently exists. For local artifact regeneration, use the isolated `local_artifacts/` command above to avoid replacing the historical files. The future independent-holdout protocol is documented in [docs/independent_holdout.md](docs/independent_holdout.md).

## Run tests

```bash
MPLCONFIGDIR=.venv/.cache/matplotlib PYTHONPATH=src:. .venv/bin/python -m pytest
```

## Run the prediction API

After installing dependencies and obtaining/configuring a compatible model, start the local service from the repository root. The default model path is `models/selected_model.joblib`, which is intentionally not included in Git. For a model generated to `local_artifacts/`, use:

```bash
LOAN_MODEL_PATH="$PWD/local_artifacts/models/selected_model.joblib" \
LOAN_BACKGROUND_PATH="$PWD/data/raw/credit_risk_dataset.csv" \
MPLCONFIGDIR=.venv/.cache/matplotlib PYTHONPATH=src:. \
  .venv/bin/python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for interactive requests. `GET /health` reports model and explanation readiness, `POST /predict` returns the default probability and threshold-based risk label, and `POST /explain` adds individual SHAP attributions. The saved model path defaults to `models/selected_model.joblib`; the local SHAP background data defaults to `data/raw/credit_risk_dataset.csv`. Override them with `LOAN_MODEL_PATH` and `LOAN_BACKGROUND_PATH` when needed. Prediction can remain available without the CSV, but `/explain` will report unavailable until compatible background data is configured. Risk labels reflect the existing validation-selected threshold and are not calibrated risk bands. SHAP contributions describe the model output and are not causal explanations.

Run API tests with:

```bash
MPLCONFIGDIR=.venv/.cache/matplotlib PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/test_api.py
```

## Run the Streamlit dashboard

Keep the FastAPI service running in one terminal, then start the dashboard from the repository root in a second terminal:

```bash
MPLCONFIGDIR=.venv/.cache/matplotlib PYTHONPATH=src:. .venv/bin/python -m streamlit run app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501` on the Overview section. Use **Settings → Appearance** in the sidebar to switch between Light and Dark mode. The choice is stored for the current Streamlit session and applies across all sections and charts. Set `LOAN_API_BASE_URL` to point at a different API host. Optional `LOAN_DATASET_PATH` and `LOAN_REPORTS_DIR` variables configure the local read-only dataset and report locations. If the API, dataset, or a report is unavailable, the related section shows an availability message; no metrics are inferred. The dashboard sends borrower input to FastAPI and does not perform model inference or preprocessing locally.

The **Overview** section shows observed dataset statistics, the selected model, validation summary, and exploratory global SHAP importance. **Risk assessment** submits the exact model feature schema to `/predict`, and optionally `/explain`. **Model analytics** separates training-only cross-validation, validation selection results, and historical preliminary test diagnostics. **Explainability** presents the saved global SHAP summary and the most recent individual explanation. Existing test metrics remain preliminary development diagnostics, not pristine final holdout results.

Run frontend logic tests with:

```bash
MPLCONFIGDIR=.venv/.cache/matplotlib PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/test_frontend.py
```

## Project layout

```text
data/raw/                 Local source CSV (ignored by Git)
notebooks/                EDA and experiment notebooks
src/loan_default/         Data validation and reusable preprocessing
api/                      FastAPI prediction and explanation service
app/                      Modular Streamlit portfolio dashboard
tests/                    Automated tests
models/                   Future model artifacts (ignored by Git)
docs/                     Data-quality and modeling decisions
```

## Prediction timing and feature policy

The prediction point is after loan terms are set and before repayment begins. The initial modeling baseline is therefore planned to include `loan_grade`, `loan_int_rate`, and `loan_percent_income`: each can be known at that point, assuming the source values are captured before repayment outcomes. They are potentially strong underwriting signals, but are not inherently leakage at this specified prediction time. The target `loan_status` is never an input.

This policy depends on confirming data provenance and timestamps. If the actual product must score applications before terms are assigned, `loan_grade` and `loan_int_rate` must be excluded; a separate sensitivity comparison without them should also be considered. Data anomalies and duplicate treatment are documented in [docs/data_quality.md](docs/data_quality.md).

## Project status and evaluation caveat

The application and local setup instructions are implemented. Deployment and an independent final holdout evaluation remain future work. Do not present the existing test metrics as a pristine final evaluation or the model as suitable for real lending decisions.
