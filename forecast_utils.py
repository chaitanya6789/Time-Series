"""
forecast_utils.py
-----------------
Shared code used by BOTH the Colab notebook (training) and the Streamlit app
(deployment).  Using one file guarantees that the features the model saw during
training are built in exactly the same way when the app makes a forecast.
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

TARGET = "PJMW_MW"
LAGS = [1, 2, 3, 24, 48, 168, 336, 720]
MAX_HISTORY = max(LAGS)          # 720 hours = 30 days of history needed

FEATURE_COLS = (
    [f"lag_{l}" for l in LAGS]
    + ["roll_mean_24", "roll_std_24", "roll_mean_168"]
    + ["Hour", "Day_of_Week", "Month", "Year", "Is_Weekend", "Is_Holiday"]
)

# US federal holidays (pandas built-in calendar, no extra package needed)
_HOLIDAYS = set(
    USFederalHolidayCalendar().holidays(start="2000-01-01", end="2045-12-31")
)


# ------------------------------------------------------------------
# 1. DATA LOADING + CLEANING
# ------------------------------------------------------------------
def clean_series(raw: pd.DataFrame, outlier_below: float = 1000.0):
    """
    Turn the raw PJMW table into a clean, strictly hourly pd.Series.

    Steps
      1. parse Datetime / numeric target, drop unreadable rows
      2. sort chronologically (the raw file is NOT sorted)
      3. average duplicated timestamps (daylight-saving 'fall back' hour)
      4. reindex to a complete hourly range (daylight-saving 'spring forward'
         hours are missing in the raw file)
      5. treat impossible values (< outlier_below MW, e.g. the 487 MW reading)
         as missing
      6. fill every gap by time interpolation
    Returns (series, report_dict)
    """
    df = raw[["Datetime", TARGET]].copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    n_raw = len(df)
    df = df.dropna(subset=["Datetime"])

    was_sorted = bool(df["Datetime"].is_monotonic_increasing)
    n_dupes = int(df["Datetime"].duplicated().sum())

    s = (df.groupby("Datetime")[TARGET].mean().sort_index())
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="h")
    n_missing_hours = int(len(full_index) - len(s))
    s = s.reindex(full_index)

    outliers = s < outlier_below
    n_outliers = int(outliers.sum())
    s[outliers] = np.nan

    n_to_fill = int(s.isna().sum())
    s = s.interpolate(method="time", limit_direction="both")
    s.index.name = "Datetime"
    s.name = TARGET

    report = dict(
        raw_rows=n_raw,
        was_sorted=was_sorted,
        duplicate_timestamps=n_dupes,
        missing_hours=n_missing_hours,
        outliers_replaced=n_outliers,
        values_filled=n_to_fill,
        clean_rows=int(len(s)),
        start=str(s.index.min()),
        end=str(s.index.max()),
    )
    return s, report


def load_clean_series(path: str):
    """Load the clean CSV written by the notebook, or the raw xlsx/csv file."""
    if path.lower().endswith(".xlsx"):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)
    return clean_series(raw)


# ------------------------------------------------------------------
# 2. FEATURES FOR TRAINING  (vectorised, whole series at once)
# ------------------------------------------------------------------
def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["Hour"] = index.hour
    out["Day_of_Week"] = index.dayofweek
    out["Month"] = index.month
    out["Year"] = index.year
    out["Is_Weekend"] = (index.dayofweek >= 5).astype(int)
    out["Is_Holiday"] = index.normalize().isin(list(_HOLIDAYS)).astype(int)
    return out


def make_feature_frame(series: pd.Series) -> pd.DataFrame:
    """Lag, rolling and calendar features for every hour of `series`."""
    X = pd.DataFrame(index=series.index)
    for l in LAGS:
        X[f"lag_{l}"] = series.shift(l)
    past = series.shift(1)                     # never include the current hour
    X["roll_mean_24"] = past.rolling(24).mean()
    X["roll_std_24"] = past.rolling(24).std()  # ddof=1, same as build_row()
    X["roll_mean_168"] = past.rolling(168).mean()
    X = X.join(_calendar_features(series.index))
    return X[FEATURE_COLS]


# ------------------------------------------------------------------
# 3. FEATURES FOR ONE FUTURE HOUR  (used in recursive forecasting / app)
# ------------------------------------------------------------------
def build_row(buf: np.ndarray, ts: pd.Timestamp) -> np.ndarray:
    """
    buf : array whose LAST element is the load of hour (ts - 1h); length >= 720
    ts  : the hour we want to predict
    """
    row = [buf[-l] for l in LAGS]
    row += [buf[-24:].mean(), buf[-24:].std(ddof=1), buf[-168:].mean()]
    row += [
        ts.hour,
        ts.dayofweek,
        ts.month,
        ts.year,
        int(ts.dayofweek >= 5),
        int(ts.normalize() in _HOLIDAYS),
    ]
    return np.asarray(row, dtype=float)


def recursive_forecast(model, history_values, next_timestamp, horizon: int):
    """
    Forecast `horizon` hours, one hour at a time.  Each prediction is appended
    to the history so it can feed the lag features of the following hour.

    history_values : 1-D array of the most recent actual loads (>= 720 values)
    next_timestamp : timestamp of the FIRST hour to predict
    """
    hist = np.asarray(history_values, dtype=float)
    if len(hist) < MAX_HISTORY:
        raise ValueError(f"Need at least {MAX_HISTORY} hours of history, got {len(hist)}.")
    buf = hist[-MAX_HISTORY:].copy()
    ts = pd.Timestamp(next_timestamp)
    preds = np.empty(horizon)
    for i in range(horizon):
        x = pd.DataFrame([build_row(buf, ts + pd.Timedelta(hours=i))], columns=FEATURE_COLS)
        p = float(model.predict(x)[0])
        p = max(p, 0.0)
        preds[i] = p
        buf = np.append(buf[1:], p)
    return preds


# ------------------------------------------------------------------
# 4. METRICS
# ------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-9)) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)
