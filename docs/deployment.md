# Deployment preparation (not deployed)

This guide prepares the existing application for Render (FastAPI) and Streamlit Community Cloud. It does not deploy either service. The repository intentionally excludes the trained model, raw CSV, and borrower-level explanation outputs.

## Runtime and dependencies

The saved training metadata records Python 3.13.1, NumPy 2.2.2, pandas 2.2.3, scikit-learn 1.9.1, and XGBoost 3.4.1. SHAP metadata records 0.53.0. `requirements.txt` pins these model-critical package versions, includes `joblib` explicitly, and installs the `src/loan_default` package from the repository's minimal `pyproject.toml`.

Use the Python 3.13 minor version on both platforms rather than assuming a particular patch release:

- **Render:** `.python-version` selects the latest supported patch release in the 3.13 line.
- **Streamlit Community Cloud:** choose Python 3.13 in the app's Advanced settings when creating it.

The package requires Python `>=3.13,<3.14`. The chosen patch should be confirmed in each platform's build/runtime logs. Exact package pins follow the recorded model-training versions; the API artifact should be smoke-checked after each deployment build.

## Render API service

Set the Render service root to the repository root.

**Build command**

```bash
pip install -r requirements.txt
```

Upload these two files through Render's Secret Files configuration, not Git:

- `selected_model.joblib.b64`: base64 text for the existing `models/selected_model.joblib` artifact.
- `shap_background.csv`: the deterministic 80-row training background with the 11 model feature columns and no `loan_status` target.

The combined content size for the current files was measured without writing either file: the model is 155,247 bytes and its base64 form is 206,996 bytes; the expected feature-only background CSV is 4,417 bytes. Combined secret-file content is **211,413 bytes**, below Render's 1 MB total limit. The CSV sample was computed in memory using the exact current API split/sample logic; no sample contents were printed or saved.

To create these private deployment files locally later, after reviewing and approving asset preparation, use commands equivalent to:

```bash
mkdir -p deployment_assets
base64 -i models/selected_model.joblib -o deployment_assets/selected_model.joblib.b64
PYTHONPATH=src .venv/bin/python - <<'PY'
from loan_default.data import FEATURES, load_data, split_data

frame = load_data("data/raw/credit_risk_dataset.csv")
training_rows = split_data(frame, random_state=42)["X_train"]
background = training_rows.sample(n=min(80, len(training_rows)), random_state=42)
background.loc[:, list(FEATURES)].to_csv(
    "deployment_assets/shap_background.csv", index=False
)
PY
```

`deployment_assets/` is ignored by Git. These commands were not run as part of this preparation.

**Render start command** (Render's shell expands `$PORT`):

```bash
base64 -d /etc/secrets/selected_model.joblib.b64 > /tmp/selected_model.joblib && \
LOAN_MODEL_PATH=/tmp/selected_model.joblib \
LOAN_BACKGROUND_PATH=/etc/secrets/shap_background.csv \
PYTHONPATH=src:. \
exec python -m uvicorn api.app:app --host 0.0.0.0 --port "$PORT"
```

The API already reads `LOAN_MODEL_PATH` and `LOAN_BACKGROUND_PATH`. The background CSV preserves the exact 80-row training-sample background used by the current API, without carrying the target column or the complete raw dataset. The API still has no authentication; anyone who can reach the deployed service can submit prediction and explanation requests. No authentication behavior was changed here.

## Streamlit Community Cloud

Select the repository, branch, and entrypoint `app/streamlit_app.py`. The repository-root `requirements.txt` and `.streamlit/config.toml` are used for this app. Configure this top-level value in the app's **Secrets** settings:

```toml
LOAN_API_BASE_URL = "https://<your-render-service>.onrender.com"
```

Streamlit Community Cloud makes top-level secrets available as environment variables, which matches the existing `os.getenv("LOAN_API_BASE_URL", ...)` configuration. No API client change is required. The dashboard reads the tracked aggregate reports and global SHAP summary from `reports/`. Since the raw CSV is excluded, dataset-derived Overview statistics will show unavailable in the cloud; the API's separately supplied feature-only background keeps individual SHAP explanations available.

The repository is private. Streamlit Community Cloud must be granted access to it, and app visibility must be configured separately from repository visibility. A private source repository does not by itself make the running dashboard private.
