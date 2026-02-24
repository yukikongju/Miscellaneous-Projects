import pandas as pd
import streamlit as st

from queries import weekly_conversions_query
from utils import get_bigquery_client, run_query
from dataclasses import dataclass
from filters import filter_segment
from plots import plot_saturation_curve


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    client = get_bigquery_client()
    df = run_query(client=client, query=weekly_conversions_query)
    return df


@dataclass()
class SegmentationSelection:
    network: str
    platform: str
    country: str
    remove_outliers: bool


def build_segment_control(df: pd.DataFrame) -> SegmentationSelection:
    network = st.selectbox("Network", options=list(df["network"].unique()), index=0)
    platform = st.selectbox("Platform", options=list(df["platform"].unique()), index=0)
    country = st.selectbox("Country", options=list(df["country"].unique()), index=0)
    remove_outliers = st.selectbox(
        "Outliers", options=[False, True], format_func=lambda x: "Yes" if x else "No", index=0
    )
    return SegmentationSelection(
        network=network, platform=platform, country=country, remove_outliers=remove_outliers
    )


st.set_page_config(page_title="Saturation Curve", page_icon="")
st.markdown("# Saturation Curves")
st.markdown(
    """
The saturation curve is computed by fitting a non-linear function (logistic
function) onto the scatterplot.

The logistic function is defined by the following formula:

$$y = L / (1 + exp(-k(x - x0)))$$

Where
- x: spend
- y: outcome (installs, paid, revenue)
- L: max attainable level (ceiling)
- k: steepness (how fast you saturate)
- x0: midpoint spend (where growth starts slowing)

The valid intervals correspond to the spend ranges that satisfies the CAC/ROAS
threshold.
"""
)

#  st.sidebar.header("Saturation Curves")

METRIC_THRESHOLD_CONFIG = {
    "installs": {"value": 5.0, "min": 0.1, "max": 50.0, "step": 0.2},
    "trials": {"value": 30.0, "min": 5.0, "max": 100.0, "step": 0.5},
    "paid": {"value": 70.0, "min": 5.0, "max": 100.0, "step": 1.0},
    "revenue": {"value": 0.4, "min": 0.0, "max": 2.0, "step": 0.01},
}


df = load_data()
segment = build_segment_control(df)
metric = st.selectbox("Metric", options=METRIC_THRESHOLD_CONFIG.keys(), index=0)
threshold = st.slider(
    #  f"{metric.title()} Threshold",
    f"{'ROAS' if metric == 'revenue' else 'CAC'} Threshold",
    min_value=METRIC_THRESHOLD_CONFIG[metric]["min"],
    max_value=METRIC_THRESHOLD_CONFIG[metric]["max"],
    value=METRIC_THRESHOLD_CONFIG[metric]["value"],
    step=METRIC_THRESHOLD_CONFIG[metric]["step"],
)

dff = filter_segment(
    df,
    network=segment.network,
    platform=segment.platform,
    country=segment.country,
    remove_outliers=segment.remove_outliers,
    x_col="spend",
    y_col=metric,
)


plot_saturation_curve(df=dff, x_col="spend", y_col=metric, threshold=threshold)
