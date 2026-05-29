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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mixpanel import load_funnel_data_from_path, load_funnel_data_from_mixpanel
from wellhub_dua import build_df_users, build_df_monthly_audit, compute_revenue

DATA_PATH = Path.home() / "Data/Wellhub/wellhub_funnel_data_20260401_20260430.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Wellhub DUA revenue audit")
    parser.add_argument("from_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("to_date", help="End date (YYYY-MM-DD)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== Loading Mixpanel funnel data ===")
    #  df_raw = load_funnel_data_from_path(str(DATA_PATH))
    df_raw = load_funnel_data_from_mixpanel(from_date=args.from_date, to_date=args.to_date)
    print(f"Raw records: {len(df_raw):,}")
    print(df_raw.head(10).to_string(index=False))
    print()

    print("=== df_users (user_id, country_code, DUA) ===")
    df_users = build_df_users(df_raw)
    print(f"Total users with at least 1 DUA: {len(df_users):,}")
    print(df_users.head(10).to_string(index=False))
    print()

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


if __name__ == "__main__":
    main()
