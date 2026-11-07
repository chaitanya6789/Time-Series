"""
train_model.py  -  rebuild the deployment files on your own computer (no Colab needed).

    python train_model.py

Reads  PJMW_MW_Hourly.xlsx  and writes
    pjm_clean.csv, xgb_model.json, metrics.json
"""
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from forecast_utils import (TARGET, FEATURE_COLS, clean_series, make_feature_frame,
                            recursive_forecast, regression_metrics)

raw = pd.read_excel("PJMW_MW_Hourly.xlsx")
series, report = clean_series(raw)
print("Cleaning report:", report)
series.reset_index().to_csv("pjm_clean.csv", index=False)

data = make_feature_frame(series).join(series).dropna()
X, y = data[FEATURE_COLS], data[TARGET]
n_train = int(len(data) * 0.8)


def new_model():
    return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8,
                        colsample_bytree=0.8, objective="reg:squarederror",
                        tree_method="hist", random_state=42, n_jobs=-1)


# 1) honest evaluation: train on the oldest 80 %, test on the newest 20 %
model = new_model().fit(X.iloc[:n_train], y.iloc[:n_train])
m1 = regression_metrics(y.iloc[n_train:], model.predict(X.iloc[n_train:]))
print("XGBoost 1-hour-ahead test metrics:", {k: round(v, 3) for k, v in m1.items()})

# 2) recursive 7-day back-test on 10 windows of the test period
pos = series.index.get_indexer(data.index)
starts = np.linspace(pos[n_train], pos[-1] - 168, 10).astype(int)
truth, pred = [], []
for p0 in starts:
    truth.append(series.values[p0:p0 + 168])
    pred.append(recursive_forecast(model, series.values[:p0], series.index[p0], 168))
truth, pred = np.array(truth), np.array(pred)
horizon = []
for h in (24, 72, 168):
    m = regression_metrics(truth[:, :h].ravel(), pred[:, :h].ravel())
    horizon.append({"Method": "XGBoost (recursive)", "Horizon (hours)": h,
                    "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE": m["MAPE"]})
print(pd.DataFrame(horizon).round(2))

# 3) final model on all data for the app
final = new_model().fit(X, y)
final.save_model("xgb_model.json")
with open("metrics.json", "w") as f:
    json.dump({"model_comparison": [{"Model": "XGBoost", **m1}], "horizon_backtest": horizon,
               "data_report": report, "deployed_model": "XGBoost (retrained on all data)"},
              f, indent=2, default=float)
print("Saved pjm_clean.csv, xgb_model.json, metrics.json  ->  now run:  streamlit run app.py")
