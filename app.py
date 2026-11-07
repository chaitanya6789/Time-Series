"""
PJM electricity load forecast - Streamlit app
Run:  streamlit run app.py
"""
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from forecast_utils import (
    FEATURE_COLS, MAX_HISTORY, TARGET,
    clean_series, recursive_forecast, regression_metrics,
)

st.set_page_config(page_title="PJM Load Forecast", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

AMBER, SLATE, RED, TEAL = "#f5a524", "#5b6b8c", "#ff6b6b", "#2dd4bf"
INK, PANEL, MUTED = "#111a2e", "#111a2e", "#6b7794"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, .stApp {{ font-family: 'IBM Plex Sans', sans-serif; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; }}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
#MainMenu, footer {{ visibility: hidden; }}
.title {{ font-family:'Space Grotesk',sans-serif; font-size:2.6rem; font-weight:700; line-height:1.05; color:{INK}; }}
.title span {{ color:{AMBER}; }}
.lede {{ margin-top:10px; color:{MUTED}; font-size:1.02rem; max-width:520px; line-height:1.55; }}
.pulse-label {{ color:{MUTED}; font-size:.86rem; margin-bottom:4px; }}
.pulse-value {{ font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:600; color:{INK}; }}
.pulse-value small {{ font-size:.95rem; color:{MUTED}; font-weight:500; }}
.strip {{ display:grid; grid-auto-flow:column; grid-auto-columns:1fr; background:{PANEL};
          border-radius:10px; margin:10px 0 18px 0; }}
.strip .cell {{ padding:16px 20px; border-right:1px solid #1f2a44; }}
.strip .cell:last-child {{ border-right:none; }}
.strip .lab {{ color:#93a1bd; font-size:.86rem; }}
.strip .val {{ font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:600; color:#f4f7fd; margin-top:2px; }}
.strip .sub {{ color:#7f8fb0; font-size:.8rem; margin-top:2px; }}
.strip .cell.hot .val {{ color:{AMBER}; }}
.sec {{ font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:600; color:{INK}; margin:22px 0 4px 0; }}
.sec-note {{ color:{MUTED}; font-size:.9rem; margin-bottom:8px; }}
.banner {{ border-left:3px solid {AMBER}; background:{PANEL}; padding:12px 16px; border-radius:0 8px 8px 0; color:#e6ebf5; margin:6px 0 14px 0; }}
.banner b {{ color:{AMBER}; }}
.side-card {{ background:{PANEL}; border-radius:8px; padding:10px 12px; margin:10px 0 14px 0; font-size:.92rem; color:#e6ebf5; }}
.side-card b {{ color:{AMBER}; }}
div.stButton > button[kind="primary"] {{ background:{AMBER}; color:#0b1220; border:1px solid {AMBER};
          font-weight:600; border-radius:8px; height:3rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def wide(fn, *args, **kwargs):
    """Call a Streamlit function full-width on old AND new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def strip(items):
    cells = "".join(
        f'<div class="cell{" hot" if hot else ""}"><div class="lab">{a}</div>'
        f'<div class="val">{b}</div><div class="sub">{c}</div></div>'
        for a, b, c, hot in items)
    st.markdown(f'<div class="strip">{cells}</div>', unsafe_allow_html=True)


def section(title, note=None):
    html = f'<div class="sec">{title}</div>'
    if note:
        html += f'<div class="sec-note">{note}</div>'
    st.markdown(html, unsafe_allow_html=True)


def sparkline(values, w=420, h=70):
    v = np.asarray(values, float)
    xs = np.linspace(2, w - 6, len(v))
    ys = h - 8 - (v - v.min()) / (v.max() - v.min() + 1e-9) * (h - 16)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"2,{h} {pts} {w - 6},{h}"
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none">'
            f'<polygon points="{area}" fill="{AMBER}" fill-opacity="0.12"/>'
            f'<polyline points="{pts}" fill="none" stroke="{AMBER}" stroke-width="1.6"/>'
            f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="3.5" fill="{AMBER}"/></svg>')


def style(chart, height=380):
    return (chart.properties(height=height)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#e6eaf2", labelFontSize=12)
            .configure_legend(orient="top", labelFontSize=12, title=None))


# ---------------------------------------------------------------- loading
HERE = os.path.dirname(os.path.abspath(__file__))


def path(name):
    return os.path.join(HERE, name)


@st.cache_resource(show_spinner="Loading XGBoost model...")
def load_model():
    try:
        from xgboost import XGBRegressor
    except ImportError:
        st.error("xgboost is not installed. Run:  pip install xgboost\n\n"
                 "On macOS also run:  brew install libomp")
        st.stop()
    except Exception as e:  # e.g. libomp missing on macOS
        st.error(f"xgboost could not be loaded: {e}\n\nOn macOS run:  brew install libomp")
        st.stop()
    model_path = path("xgb_model.json")
    if os.path.exists(model_path):
        try:
            model = XGBRegressor()
            model.load_model(model_path)
            return model
        except Exception:
            # saved file is unreadable (different XGBoost version / corrupted upload)
            # -> ignore it and train a fresh model below
            pass
    # No saved model in the repo -> train it once (about 1-2 minutes) and save it
    from forecast_utils import make_feature_frame
    with st.spinner("First start: training the XGBoost model (about 1-2 minutes, only happens once)..."):
        s = load_data()
        data = make_feature_frame(s).join(s).dropna()
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8,
                             colsample_bytree=0.8, objective="reg:squarederror",
                             tree_method="hist", random_state=42, n_jobs=-1)
        model.fit(data[FEATURE_COLS], data[TARGET])
        try:
            model.save_model(model_path)
        except Exception:
            pass  # read-only disk: model simply stays in memory
    return model


@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_data():
    if os.path.exists(path("pjm_clean.csv")):
        d = pd.read_csv(path("pjm_clean.csv"), parse_dates=["Datetime"])
        return d.set_index("Datetime")[TARGET].asfreq("h")
    raw = pd.read_excel(path("PJMW_MW_Hourly.xlsx"))
    s, _ = clean_series(raw)
    return s


@st.cache_data
def load_metrics():
    p = path("metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


try:
    model = load_model()
    series = load_data()
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}. Keep app.py, forecast_utils.py, xgb_model.json "
             "and pjm_clean.csv (or PJMW_MW_Hourly.xlsx) in the same folder.")
    st.stop()

metrics = load_metrics()
first_ts, last_ts = series.index.min(), series.index.max()
last_load = float(series.iloc[-1])

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("## Forecast settings")
days = st.sidebar.slider("Forecast period (days)", 1, 30, 7)
hours = days * 24
st.sidebar.markdown(f'<div class="side-card"><b>{days} day(s)</b> = <b>{hours}</b> hourly predictions</div>',
                    unsafe_allow_html=True)

st.sidebar.markdown("### Forecast start")
after_end = st.sidebar.checkbox("Start right after the last dataset value", value=True)
if after_end:
    start = last_ts + pd.Timedelta(hours=1)
else:
    min_date = (first_ts + pd.Timedelta(hours=MAX_HISTORY)).date()
    max_date = (last_ts + pd.Timedelta(days=30)).date()
    d = st.sidebar.date_input("Select date", value=last_ts.date() - pd.Timedelta(days=60),
                              min_value=min_date, max_value=max_date)
    t = st.sidebar.time_input("Select time", value=pd.Timestamp("00:00").time())
    start = pd.Timestamp(f"{d} {t}").floor("h")
    start = max(start, first_ts + pd.Timedelta(hours=MAX_HISTORY))
st.sidebar.markdown(f'<div class="side-card">Forecast starts at<br><b>{start:%Y-%m-%d %H:%M}</b></div>',
                    unsafe_allow_html=True)
if not after_end and start <= last_ts:
    st.sidebar.info("Start is inside the dataset, so the forecast is compared with the real load (back-test).")

with st.sidebar.expander("About the model"):
    st.markdown("- XGBoost Regressor\n- Recursive hourly forecasting\n"
                "- Lags: 1, 2, 3, 24, 48, 168, 336, 720 h\n- Rolling mean / std\n"
                "- Hour, weekday, month, year, weekend, US holiday")

# ---------------------------------------------------------------- header
c1, c2 = st.columns([1.25, 1])
with c1:
    st.markdown('<div class="title">PJM electricity<br>load <span>forecast</span></div>'
                '<div class="lede">Hourly demand forecasts for up to 30 days ahead, generated by a '
                'trained XGBoost model on PJM West data.</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="pulse-label">Last 7 days of actual load</div>{sparkline(series.tail(168).values)}'
                f'<div class="pulse-value">{last_load:,.0f} <small>MW at {last_ts:%Y-%m-%d %H:%M}</small></div>',
                unsafe_allow_html=True)

strip([("Historical data start", f"{first_ts:%Y-%m-%d}", "First record", False),
       ("Last available data", f"{last_ts:%Y-%m-%d %H:%M}", "Latest record", False),
       ("Hourly records", f"{len(series):,}", "After cleaning", False),
       ("Last actual load", f"{last_load:,.0f} MW", "Latest reading", True)])

# ---------------------------------------------------------------- run
if wide(st.button, f"Generate {days}-day forecast", type="primary"):
    with st.spinner(f"Generating {hours} hourly predictions..."):
        if start <= last_ts:                        # back-test inside the data
            pos = series.index.get_loc(start)
            history = series.values[:pos]
            preds = recursive_forecast(model, history, start, hours)
            idx = pd.date_range(start, periods=hours, freq="h")
            hist_df = series.iloc[max(0, pos - 168):pos]
            actual = series.reindex(idx)
        else:                                       # true future forecast
            gap = int((start - last_ts) / pd.Timedelta(hours=1)) - 1
            full = recursive_forecast(model, series.values, last_ts + pd.Timedelta(hours=1), gap + hours)
            preds = full[gap:]
            idx = pd.date_range(start, periods=hours, freq="h")
            hist_df = series.tail(168)
            actual = pd.Series(np.nan, index=idx)

    forecast_df = pd.DataFrame({"Prediction_Number": np.arange(1, hours + 1),
                                "Datetime": idx, "Predicted_MW": preds})
    daily_df = (forecast_df.set_index("Datetime")["Predicted_MW"].resample("D")
                .agg(Average_MW="mean", Maximum_MW="max", Minimum_MW="min", Total_MWh="sum").reset_index())
    st.session_state["result"] = dict(forecast_df=forecast_df, daily_df=daily_df, hist_df=hist_df,
                                      actual=actual, start=start, days=days, hours=hours)

res = st.session_state.get("result")
if res is None:
    st.info("**1.** Pick a period in the sidebar  →  **2.** choose the start  →  **3.** click *Generate*.")
    if metrics:
        st.caption("Scroll down after generating a forecast to see model accuracy.")
    st.stop()

fdf, ddf, hist_df, actual = res["forecast_df"], res["daily_df"], res["hist_df"], res["actual"]
preds = fdf["Predicted_MW"].values
times = fdf["Datetime"]
imax, imin = int(np.argmax(preds)), int(np.argmin(preds))

st.markdown(f'<div class="banner">Forecast ready: <b>{res["hours"]} hourly predictions</b> for '
            f'{res["days"]} day(s), starting {res["start"]:%Y-%m-%d %H:%M}.</div>', unsafe_allow_html=True)
strip([("Average load", f"{preds.mean():,.0f} MW", f"Across {res['hours']} hours", False),
       ("Peak load", f"{preds.max():,.0f} MW", f"{times.iloc[imax]:%Y-%m-%d %H:%M}", True),
       ("Minimum load", f"{preds.min():,.0f} MW", f"{times.iloc[imin]:%Y-%m-%d %H:%M}", False),
       ("Total energy", f"{preds.sum() / 1000:,.1f} GWh", "Sum of hourly load", False)])

has_actual = actual.notna().all()
if has_actual:
    m = regression_metrics(actual.values, preds)
    strip([("Back-test MAE", f"{m['MAE']:,.0f} MW", "vs real load", False),
           ("Back-test RMSE", f"{m['RMSE']:,.0f} MW", "vs real load", False),
           ("Back-test MAPE", f"{m['MAPE']:.2f} %", "vs real load", True),
           ("Back-test R²", f"{m['R2']:.3f}", "vs real load", False)])

tabs = st.tabs(["Forecast", "Daily view", "Hourly data", "Export", "Model performance"])

# ---- Forecast tab
with tabs[0]:
    section("Hourly load forecast", "Slate = last 7 days of actual load, amber = forecast"
            + (", teal = real load during the forecast window." if has_actual else "."))
    parts = [hist_df.rename(TARGET).reset_index().rename(columns={TARGET: "Load_MW"}).assign(Series="Recent actual"),
             fdf[["Datetime", "Predicted_MW"]].rename(columns={"Predicted_MW": "Load_MW"}).assign(Series="Forecast")]
    domain, colors = ["Recent actual", "Forecast"], [SLATE, AMBER]
    if has_actual:
        parts.append(pd.DataFrame({"Datetime": actual.index, "Load_MW": actual.values}).assign(Series="Actual (back-test)"))
        domain.append("Actual (back-test)"); colors.append(TEAL)
    comb = pd.concat(parts, ignore_index=True)
    base = alt.Chart(comb).encode(
        x=alt.X("Datetime:T", title=None),
        y=alt.Y("Load_MW:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=alt.Scale(domain=domain, range=colors)))
    layers = [base.mark_line(strokeWidth=2.2),
              base.mark_circle(size=50, opacity=0).encode(
                  tooltip=["Series:N", alt.Tooltip("Datetime:T", format="%Y-%m-%d %H:%M"),
                           alt.Tooltip("Load_MW:Q", format=",.1f")])]
    marks = pd.DataFrame({"Datetime": [times.iloc[imax], times.iloc[imin]], "Load_MW": [preds[imax], preds[imin]],
                          "Label": ["Peak", "Low"]})
    layers += [alt.Chart(marks).mark_circle(size=130).encode(
        x="Datetime:T", y="Load_MW:Q", color=alt.Color("Label:N", scale=alt.Scale(domain=["Peak", "Low"], range=[RED, TEAL]), legend=None))]
    layers += [alt.Chart(pd.DataFrame({"Datetime": [res["start"]]})).mark_rule(strokeDash=[5, 4], color=MUTED).encode(x="Datetime:T")]
    wide(st.altair_chart, style(alt.layer(*layers).interactive(), 400), theme=None)
    st.caption("Forecasts are recursive: every predicted hour becomes part of the history used to build "
               "the lag features of the next hour, so uncertainty grows with the horizon.")

# ---- Daily tab
with tabs[1]:
    section("Daily average load", "Bars = daily average, thin line = daily min to max.")
    bars = alt.Chart(ddf).mark_bar(color=AMBER, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Datetime:T", timeUnit="yearmonthdate", title=None),
        y=alt.Y("Average_MW:Q", title="Load (MW)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Datetime:T", format="%Y-%m-%d"), alt.Tooltip("Average_MW:Q", format=",.1f"),
                 alt.Tooltip("Maximum_MW:Q", format=",.1f"), alt.Tooltip("Minimum_MW:Q", format=",.1f")])
    rng = alt.Chart(ddf).mark_rule(color=INK, strokeWidth=1.5).encode(
        x=alt.X("Datetime:T", timeUnit="yearmonthdate"), y="Minimum_MW:Q", y2="Maximum_MW:Q")
    wide(st.altair_chart, style(alt.layer(bars, rng), 320), theme=None)
    if res["days"] >= 2:
        section("Load pattern by hour", "Each row is a day, each column an hour. Brighter = higher demand.")
        heat_df = fdf.assign(Date=fdf["Datetime"].dt.strftime("%b %d"), Hour=fdf["Datetime"].dt.hour)
        heat = alt.Chart(heat_df).mark_rect().encode(
            x=alt.X("Hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Date:O", sort=None, title=None),
            color=alt.Color("Predicted_MW:Q", scale=alt.Scale(scheme="inferno"), legend=alt.Legend(title="MW", orient="right")),
            tooltip=["Date:O", "Hour:O", alt.Tooltip("Predicted_MW:Q", format=",.1f")])
        wide(st.altair_chart, style(heat, max(180, 26 * res["days"] + 60)), theme=None)
    section("Daily summary table")
    wide(st.dataframe, ddf, hide_index=True)

# ---- Hourly tab
with tabs[2]:
    section(f"Hourly predictions ({res['hours']} rows)")
    show = fdf.copy()
    if has_actual:
        show["Actual_MW"] = actual.values
        show["Abs_Error_MW"] = (show["Actual_MW"] - show["Predicted_MW"]).abs()
    wide(st.dataframe, show, hide_index=True)

# ---- Export tab
with tabs[3]:
    section("Download results", "Both files open directly in Excel.")
    a, b = st.columns(2)
    with a:
        wide(st.download_button, "Download hourly forecast (CSV)", fdf.to_csv(index=False).encode(),
             file_name=f"pjm_load_forecast_{res['days']}_days.csv", mime="text/csv")
    with b:
        wide(st.download_button, "Download daily summary (CSV)", ddf.to_csv(index=False).encode(),
             file_name=f"pjm_daily_summary_{res['days']}_days.csv", mime="text/csv")

# ---- Model performance tab
with tabs[4]:
    if not metrics:
        st.info("metrics.json not found. Run the Colab notebook to generate it.")
    else:
        section("Model comparison (1-hour-ahead, chronological test set)")
        wide(st.dataframe, pd.DataFrame(metrics["model_comparison"]).round(3), hide_index=True)
        if metrics.get("horizon_backtest"):
            section("Recursive back-test (what this app really does)",
                    "Average error of XGBoost when it forecasts many hours ahead, across several test-period windows.")
            wide(st.dataframe, pd.DataFrame(metrics["horizon_backtest"]).round(3), hide_index=True)
