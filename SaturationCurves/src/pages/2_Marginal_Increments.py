import pandas as pd
import streamlit as st

from filters import get_marginal_spend_metric_table
from queries import weekly_conversions_query
from utils import get_bigquery_client, run_query


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    client = get_bigquery_client()
    df = run_query(client=client, query=weekly_conversions_query)
    return df


df = load_data()

st.markdown("# Marginal Increments")
st.markdown(
    """

Marginal increments show what you get from the next dollar of spend, not the average of past spend. For each channel segment (network,
  platform, country), we fit a saturation curve and estimate incremental outcomes (installs, paid users, revenue) at different spend percentile.
"""
)

st.markdown("### Marginal CAC Increments")

df_marginal_cac = get_marginal_spend_metric_table(df=df, spend_col="spend", metric_col="paid")
st.dataframe(df_marginal_cac)

st.markdown("### Marginal ROAS Increments")

df_marginal_roas = get_marginal_spend_metric_table(df=df, spend_col="spend", metric_col="revenue")
st.dataframe(df_marginal_roas)
