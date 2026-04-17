import pandas as pd
import streamlit as st
import plotly.express as px
from dataclasses import dataclass

from queries import daily_t2p_comparison
from utils import get_bigquery_client, run_query


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """Load weekly conversions data from BigQuery with caching.

    Returns:
        Weekly conversions data.
    """
    client = get_bigquery_client()
    df = run_query(client=client, query=daily_t2p_comparison)
    return df


@dataclass()
class SegmentationSelection:
    """Container for selected segment dimensions and filter options."""

    network: str
    platform: str
    country: str


def build_segment_control(df: pd.DataFrame) -> SegmentationSelection:
    """Render Streamlit controls for segment selection.

    Args:
        df: Source DataFrame containing segment dimension columns.

    Returns:
        Selected segment configuration.
    """
    network = st.selectbox("Network", options=list(df["network"].unique()), index=0)
    platform = st.selectbox("Platform", options=list(df["platform"].unique()), index=0)
    country = st.selectbox("Country", options=list(df["country"].unique()), index=0)
    return SegmentationSelection(
        network=network,
        platform=platform,
        country=country,
    )


st.set_page_config(page_title="Daily T2P Comparison", page_icon="")
st.markdown("# Daily T2P Comparison")
st.markdown(
    """
Breakdown Definitions:
- `current`: Value in our current pipeline. T2P is computed using a moving
  average and selects either real T2P or modeled T2P based on the `need_modeling` flag.
- `partners_daily`: Values computed using selected partners data (ie data we get from API). T2P is computed at a daily cohorted granularity
- `backend_daily`: Values computed using backend data (ie data we get from app events). T2P is computed at a daily cohorted granularity
"""
)

# fetching the data
df = load_data()
# st.write(df)
segment = build_segment_control(df)

# filtering data based on selected segment
mask = (
    (df["network"] == segment.network)
    & (df["platform"] == segment.platform)
    & (df["country"] == segment.country)
)
dff = df[mask]

#
for metric in [
    "t2p",
    "paid",
    "revenue",
]:  #  "rev_per_paid"
    fig = px.line(
        dff,
        x="date",
        y=metric,
        color="source",
        markers=True,
        title=f"{metric.capitalize()} by Source",
    )
    st.plotly_chart(fig, use_container_width=True)
