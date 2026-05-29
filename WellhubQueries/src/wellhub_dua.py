import pandas as pd

DUA_RATE = {
    "US": 1.75,
    "GB": 1.75,
    "IE": 1.75,
    "DE": 1.75,
    "ES": 1.75,
    "IT": 1.75,
    "BR": 1.50,
    "MX": 1.25,
    "CL": 1.25,
    "AR": 1.25,
}
DEFAULT_RATE = 1.25
DUA_CAP = 5


def build_df_users(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw daily events into per-user monthly DUA count (uncapped).

    A DUA is one distinct day where create_record_wellhub == 1.
    Returns columns: user_id, country_code, DUA
    """
    df_active = df_raw[df_raw["create_record_wellhub"] == 1]
    df_users = (
        df_active.groupby(["user_id", "country_code"]).agg(DUA=("date", "nunique")).reset_index()
    )
    return df_users[["user_id", "country_code", "DUA"]]


def build_df_monthly_audit(df_users: pd.DataFrame) -> pd.DataFrame:
    """Group users by country and capped DUA count.

    Returns columns: country_code, DUA, num_users
    """
    return (
        df_users.groupby(["country_code", "DUA"])
        .agg(num_users=("user_id", "count"))
        .reset_index()
        .sort_values(["country_code", "DUA"])
        .reset_index(drop=True)
    )


def compute_revenue(df_users: pd.DataFrame, bill_unmapped: bool = False) -> float:
    """Compute total monthly revenue from capped DUA counts.

    bill_unmapped=True  → unmapped countries use DEFAULT_RATE ($1.25)
    bill_unmapped=False → unmapped countries are excluded ($0)
    """
    fallback = DEFAULT_RATE if bill_unmapped else 0.0
    capped_dua = df_users["DUA"].clip(upper=DUA_CAP)
    rates = df_users["country_code"].map(DUA_RATE).fillna(fallback)
    return (capped_dua * rates).sum()
