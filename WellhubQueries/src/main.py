"""
Wellhub DUA Revenue Audit
=========================
Fetches daily Wellhub engagement data from the Mixpanel funnel API and computes
monthly revenue based on the Daily User Action (DUA) billing model.

DUA Logic
---------
Each day a Wellhub subscriber engages with BetterSleep counts as one DUA,
regardless of how many times they open the app that day. DUAs are capped at 5
per user per month for billing purposes.

Rates by region:
  $1.75/DUA — US, GB, IE, DE, ES, IT  (max $8.75/user/month)
  $1.50/DUA — BR                       (max $7.50/user/month)
  $1.25/DUA — MX, CL, AR              (max $6.25/user/month)
  $1.25/DUA — all other countries (if bill_unmapped=True, else $0)

Usage
-----
  python src/main.py <from_date> <to_date>

  Arguments:
    from_date   Start of the billing period (YYYY-MM-DD, inclusive)
    to_date     End of the billing period   (YYYY-MM-DD, inclusive)

  Example:
    uv run python src/main.py 2026-04-01 2026-04-30

  Credentials (MIXPANEL_USERNAME, MIXPANEL_PASSWORD) are read from .env.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from mixpanel import (
    MixpanelAPI,
    get_mixpanel_funnel_dataframe_from_json,
)
from wellhub_dua import build_df_monthly_audit, compute_revenue
from constants import COUNTRY_CODE_MAP, MAX_DUA_PER_MONTH

DATA_PATH = Path.home() / "Data/Wellhub/wellhub_funnel_data_20260401_20260430.json"

log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Wellhub DUA revenue audit")
    parser.add_argument("from_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("to_date", help="End date (YYYY-MM-DD)")
    return parser.parse_args()


def load_config() -> dict:
    """Load and validate required environment variables."""
    load_dotenv()
    required = ("MIXPANEL_PROJECT_ID", "MIXPANEL_USERNAME", "MIXPANEL_PASSWORD")
    config = {key: os.getenv(key) for key in required}
    missing = [k for k, v in config.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")
    return config


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    ## Loading system environment from dotenv files
    config = load_config()

    FUNNEL_ID = 90364672  # TODO: add mixpanel link

    ## Data Extraction from Mixpanel
    print("=== Loading Mixpanel funnel data ===")
    mixpanel_api = MixpanelAPI(
        project_id=config["MIXPANEL_PROJECT_ID"],
        username=config["MIXPANEL_USERNAME"],
        password=config["MIXPANEL_PASSWORD"],
    )
    response = mixpanel_api.query_funnel(
        funnel_id=FUNNEL_ID, from_date=args.from_date, to_date=args.to_date
    )
    if not response:
        log.error("No wellhub data from mixpanel, please check!")
        raise ValueError(
            "Empty response from Mixpanel funnel API — check funnel ID and date range."
        )

    df = get_mixpanel_funnel_dataframe_from_json(data=response)

    ## Cleanup -> (1) remove users without any user_id; (2) remove duplicates ; (3) remap country
    columns_mappings = {
        "Date YYYY-MM-DD": "date",
        "$country_code": "country_code",
        "$user_id": "user_id",
        "count": "count",
    }
    if not all(col in df.columns for col in columns_mappings.keys()):
        log.error("One or more expected columns are missing from the dataframe.")
        raise ValueError("Missing expected columns in Mixpanel data.")

    df_clean = df.rename(columns=columns_mappings)
    undefined_user_id_mask = df_clean["user_id"] == "undefined"
    df_clean = df_clean[~undefined_user_id_mask]
    df_clean = df_clean.drop_duplicates(subset=["date", "country_code", "user_id"])
    df_clean["country_code"] = df_clean["country_code"].replace(COUNTRY_CODE_MAP)

    ##
    print("=== df_users (user_id, country_code, DUA) ===")
    df_users = (
        df_clean.groupby(["user_id", "country_code"]).agg(DUA=("date", "count")).reset_index()
    )
    df_users["DUA"] = df_users["DUA"].clip(lower=0, upper=MAX_DUA_PER_MONTH)

    print("=== df_monthly_audit (country_code, DUA, num_users) ===")
    df_monthly_audit = build_df_monthly_audit(df_users)
    print(df_monthly_audit.to_string(index=False))
    print()

    total_revenue = compute_revenue(df_users, bill_unmapped=False)
    total_revenue_with_default = compute_revenue(df_users, bill_unmapped=True)
    print(f"=== Total monthly revenue (unmapped excluded):       ${total_revenue:,.2f} ===")
    print(
        f"=== Total monthly revenue (unmapped @ default rate): ${total_revenue_with_default:,.2f} ==="
    )

    ## TODO: save spreadsheets

    ## TODO: print warnings => (1) country not mapped, (2) users without user_id, (3)


if __name__ == "__main__":
    main()
