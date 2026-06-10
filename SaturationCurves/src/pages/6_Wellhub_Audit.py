"""Wellhub monthly revenue audit — DUA billing model."""

import calendar
from datetime import date, datetime

import pandas as pd
import streamlit as st

from queries import query_create_wellhub_record
from utils import get_bigquery_client, run_query

# ─── Constants ───────────────────────────────────────────────────────────────

DUA_RATE: dict[str, float] = {
    "US": 1.75,
    "GB": 1.75,
    "UK": 1.75,
    "IE": 1.75,
    "DE": 1.75,
    "ES": 1.75,
    "IT": 1.75,
    "RO": 1.75,
    "BR": 1.50,
    "MX": 1.25,
    "CL": 1.25,
    "AR": 1.25,
}
DEFAULT_RATE = 1.25
DUA_CAP = 5

COUNTRY_CODE_MAP: dict[str, str] = {
    "United States": "US",
    "United Kingdom": "GB",
    "Ireland": "IE",
    "Germany": "DE",
    "Spain": "ES",
    "Italy": "IT",
    "Brazil": "BR",
    "Mexico": "MX",
    "Chile": "CL",
    "Argentina": "AR",
    "Canada": "CA",
    "Guatemala": "GT",
    "Belgium": "BE",
    "Martinique": "MQ",
    "Singapore": "SG",
    "France": "FR",
    "Australia": "AU",
    "Colombia": "CO",
    "Peru": "PE",
    "Portugal": "PT",
    "Netherlands": "NL",
    "Sweden": "SE",
    "Norway": "NO",
    "Denmark": "DK",
    "Finland": "FI",
    "Switzerland": "CH",
    "Austria": "AT",
    "Poland": "PL",
    "Japan": "JP",
    "South Korea": "KR",
    "India": "IN",
    "New Zealand": "NZ",
    "South Africa": "ZA",
    "Greece": "GR",
    "Hong Kong": "HK",
    "Kenya": "KE",
    "Puerto Rico": "PR",
    "Albania": "AL",
    "Aruba": "AW",
    "Cabo Verde": "CV",
    "Costa Rica": "CR",
    "Romania": "RO",
}


# ─── Data loading ────────────────────────────────────────────────────────────


@st.cache_data(ttl=4 * 3600)
def load_raw_data() -> pd.DataFrame:
    client = get_bigquery_client()
    df = run_query(client=client, query=query_create_wellhub_record)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─── Transformation helpers ──────────────────────────────────────────────────


def get_previous_months(n: int = 3) -> list[tuple[int, int]]:
    """Return the last n complete (year, month) tuples, most recent first."""
    today = date.today()
    y, m = today.year, today.month - 1
    if m == 0:
        m, y = 12, y - 1
    result = []
    for _ in range(n):
        result.append((y, m))
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return result


def filter_month(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    mask = (df["date"].dt.year == year) & (df["date"].dt.month == month)
    return df[mask].copy()


def build_df_users(df_month: pd.DataFrame) -> pd.DataFrame:
    """Clean raw monthly data and return per-user capped DUA counts."""
    df = df_month.copy()
    df["country_code"] = df["country_code"].replace(COUNTRY_CODE_MAP)
    df = df[df["user_id"] != "undefined"]
    df = df.drop_duplicates(subset=["date", "country_code", "user_id"])
    df_users = df.groupby(["user_id", "country_code"]).agg(DUA=("date", "count")).reset_index()
    df_users["DUA"] = df_users["DUA"].clip(lower=0, upper=DUA_CAP)

    ## FIXED: remove users who are mapped to multiple countries
    #  df_users = df_users.sort_values(by=["user_id", "DUA"], ascending=[True, False])
    #  df_users = df_users.drop_duplicates(subset="user_id")

    ## FIX: only keep user whose user_id starts with '$device:'
    #  device_mask = df_users["user_id"].str.startswith("$device:")
    #  df_users = df_users[device_mask]

    return df_users


def build_df_monthly_audit(df_users: pd.DataFrame) -> pd.DataFrame:
    """Group by country and DUA tier to show user counts per bucket."""
    return (
        df_users.groupby(["country_code", "DUA"])
        .agg(num_users=("user_id", "count"))
        .reset_index()
        .sort_values(["country_code", "DUA"])
        .reset_index(drop=True)
    )


def compute_revenue(df_users: pd.DataFrame, bill_unmapped: bool = False) -> float:
    fallback = DEFAULT_RATE if bill_unmapped else 0.0
    capped_dua = df_users["DUA"].clip(upper=DUA_CAP)
    rates = df_users["country_code"].map(DUA_RATE).fillna(fallback)
    return float((capped_dua * rates).sum())


def enrich_audit(df_audit: pd.DataFrame) -> pd.DataFrame:
    """Add type and estimated revenue columns to the audit table."""
    df = df_audit.copy()
    df["type"] = df["country_code"].apply(lambda c: "mapped" if c in DUA_RATE else "unmapped")
    rates = df["country_code"].map(DUA_RATE).fillna(DEFAULT_RATE)
    df["est_revenue"] = df["DUA"] * rates * df["num_users"]
    return (
        df[["country_code", "type", "DUA", "num_users", "est_revenue"]]
        .sort_values("est_revenue", ascending=False)
        .reset_index(drop=True)
    )


# ─── Page ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Wellhub Audit", page_icon="📊")

st.markdown("# 📊 Wellhub revenue audit")
st.caption(f"Last refreshed: today at {datetime.now().strftime('%I:%M %p')} · Cache TTL 4h")

# Load data
with st.spinner("Loading Wellhub data…"):
    df_raw = load_raw_data()

# ── Month selector ────────────────────────────────────────────────────────────
months = get_previous_months(3)
month_labels = [f"{calendar.month_abbr[m]} {y}" for y, m in months]

col_period, col_clear = st.columns([3, 1])

with col_period:
    selected_idx = st.radio(
        "Period",
        options=range(len(months)),
        format_func=lambda i: month_labels[i],
        index=0,
        horizontal=True,
    )

with col_clear:
    st.write("")
    if st.button("🔄 Clear cache", key="clear_cache_btn"):
        st.cache_data.clear()
        st.rerun()

selected_year, selected_month = months[selected_idx]
selected_label = month_labels[selected_idx]
#  print(selected_year, selected_month)

# ── Compute selected month ────────────────────────────────────────────────────
df_month = filter_month(df_raw, selected_year, selected_month)

# Debug info
with st.expander("🐛 Debug: Data integrity checks"):
    st.write(f"**Raw month data rows:** {len(df_month)}")

    undefined_count = (df_month["user_id"] == "undefined").sum()
    st.write(f"**'undefined' user_id rows:** {undefined_count}")

    df_after_undefined = df_month[df_month["user_id"] != "undefined"]
    st.write(f"**Rows after filtering undefined:** {len(df_after_undefined)}")

    df_after_dedup = df_after_undefined.drop_duplicates(subset=["date", "country_code", "user_id"])
    st.write(f"**Rows after dedup (date, country, user):** {len(df_after_dedup)}")
    st.write(f"**Unique users in month:** {df_after_dedup['user_id'].nunique()}")

    unmapped_countries = df_month[~df_month["country_code"].isin(COUNTRY_CODE_MAP.keys())][
        "country_code"
    ].unique()
    unmapped_countries = [c for c in unmapped_countries if c is not None]
    if len(unmapped_countries) > 0:
        st.warning(f"**Unmapped countries in data:** {sorted(unmapped_countries)}")

df_users = build_df_users(df_month)
df_audit = build_df_monthly_audit(df_users)

revenue_mapped = compute_revenue(df_users, bill_unmapped=False)
revenue_combined = compute_revenue(df_users, bill_unmapped=True)
revenue_unmapped = revenue_combined - revenue_mapped

## TODO: some users mapped to multiple countries

# Delta vs previous month
delta_str: str | None = None
if selected_idx + 1 < len(months):
    prev_year, prev_month = months[selected_idx + 1]
    df_prev_month = filter_month(df_raw, prev_year, prev_month)
    df_users_prev = build_df_users(df_prev_month)
    revenue_prev = compute_revenue(df_users_prev, bill_unmapped=False)
    if revenue_prev:
        pct = (revenue_mapped - revenue_prev) / revenue_prev * 100
        delta_str = f"{pct:+.1f}% vs {calendar.month_abbr[prev_month]}"

# ── Revenue summary ───────────────────────────────────────────────────────────
st.markdown(f"#### Revenue summary — {selected_label}")

col1, col2, col3 = st.columns(3)
col1.metric("Mapped revenue", f"${revenue_mapped:,.0f}", delta=delta_str)
col2.metric(
    "Unmapped revenue",
    f"${revenue_unmapped:,.0f}",
    delta="@ default rate",
    delta_color="off",
)
col3.metric(
    "Combined estimate",
    f"${revenue_combined:,.0f}",
    delta="mapped + unmapped",
    delta_color="off",
)

# ── Invoice comparison ────────────────────────────────────────────────────────
col_inv, col_disc = st.columns(2)
with col_inv:
    with st.container(border=True):
        invoice = st.number_input(
            "Wellhub invoice ($)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
            help="Enter the amount from the Wellhub invoice to compute the discrepancy.",
        )

with col_disc:
    if invoice > 0:
        discrepancy = revenue_combined - invoice
        disc_pct = (discrepancy / invoice) * 100
        col_disc.metric(
            "Discrepancy",
            f"${discrepancy:+,.0f}",
            delta=f"{disc_pct:+.2f}% vs invoice",
            delta_color="inverse",
        )
    else:
        col_disc.metric("Discrepancy", "—", help="Enter the Wellhub invoice amount above.")

# ── Data tables ───────────────────────────────────────────────────────────────
tab_users, tab_audit, tab_by_country = st.tabs(
    ["Users table · df_users", "Audit table · df_monthly_audit", "Revenue by country"]
)

with tab_users:
    search_users = st.text_input("Search user_id or country…", key="user_search")
    df_display_users = df_users.copy()
    if search_users:
        mask = df_display_users["user_id"].str.contains(
            search_users, case=False, na=False
        ) | df_display_users["country_code"].str.contains(search_users, case=False, na=False)
        df_display_users = df_display_users[mask]
    st.dataframe(df_display_users, use_container_width=True, height=300)

with tab_audit:
    search_audit = st.text_input("Search country_code…", key="audit_search")
    df_display_audit = enrich_audit(df_audit)
    if search_audit:
        df_display_audit = df_display_audit[
            df_display_audit["country_code"].str.contains(search_audit, case=False, na=False)
        ]
    df_display_audit["est_revenue"] = df_display_audit["est_revenue"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(df_display_audit, use_container_width=True, height=300)

with tab_by_country:
    df_by_country = (
        df_audit.copy()
        .assign(rate=lambda d: d["country_code"].map(DUA_RATE).fillna(DEFAULT_RATE))
        .assign(revenue=lambda d: d["DUA"] * d["rate"] * d["num_users"])
        .groupby("country_code")
        .agg(
            num_users=("num_users", "sum"),
            total_dua_days=(
                "DUA",
                lambda x: (x * df_audit.loc[x.index, "num_users"]).sum(),
            ),
            revenue=("revenue", "sum"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    df_by_country["revenue"] = df_by_country["revenue"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(df_by_country, use_container_width=True)

# ── Raw data inspection ───────────────────────────────────────────────────────
with st.expander("📋 Raw data inspection"):
    st.write(f"**Sample of raw month data ({len(df_month)} rows):**")
    st.dataframe(df_month.head(50), use_container_width=True, height=400)

# ── Downloads ─────────────────────────────────────────────────────────────────
st.markdown("#### Downloads")
month_str = f"{selected_year}_{selected_month:02d}"
col_dl1, col_dl2 = st.columns(2)

col_dl1.download_button(
    label=f"⬇ wellhub_users_{month_str}.csv",
    data=df_users.to_csv(index=False).encode("utf-8"),
    file_name=f"wellhub_users_{month_str}.csv",
    mime="text/csv",
)
col_dl2.download_button(
    label=f"⬇ wellhub_audit_{month_str}.csv",
    data=df_audit.to_csv(index=False).encode("utf-8"),
    file_name=f"wellhub_audit_{month_str}.csv",
    mime="text/csv",
)
