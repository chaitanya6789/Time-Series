# PJM Hourly Electricity Load Forecasting (Mini Project)

Time-series forecasting of hourly PJM West demand (MW) with a Streamlit web app.

## Files
| File | Purpose |
|---|---|
| `PJM_Load_Forecasting.ipynb` | Colab notebook: cleaning, EDA, features, 7 models, evaluation, saving |
| `forecast_utils.py` | Cleaning + feature code shared by notebook and app |
| `app.py` | Streamlit app |
| `xgb_model.json` | Trained XGBoost model (portable JSON, no pickle problems) |
| `pjm_clean.csv` | Cleaned dataset used by the app |
| `metrics.json` | Model scores shown in the app |
| `train_model.py` | Optional: rebuild model + clean data on your own computer (no Colab) |
| `requirements.txt` | Minimal dependencies for the app |

## Run on macOS (Apple Silicon or Intel)
```bash
# 1. one-time: OpenMP library that XGBoost needs on macOS
brew install libomp

# 2. create a clean environment (Python 3.10 - 3.12)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. run
streamlit run app.py
```
Opens at http://localhost:8501

## Deploy on Streamlit Community Cloud
Push `app.py`, `forecast_utils.py`, `xgb_model.json`, `pjm_clean.csv`, `metrics.json`,
`requirements.txt` and `.streamlit/config.toml` to a GitHub repo, then create the app at share.streamlit.io.

## Rebuild the model locally (optional, instead of Colab)
Put `PJMW_MW_Hourly.xlsx` in this folder, then:
```bash
python train_model.py      # writes pjm_clean.csv, xgb_model.json, metrics.json
streamlit run app.py
```
