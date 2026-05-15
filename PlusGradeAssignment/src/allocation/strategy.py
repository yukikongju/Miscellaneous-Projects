"""
Offer scoring, greedy allocation, and business-constraint enforcement.

Usage:
    from allocation.strategy import score_offers, greedy_allocate, enforce_price_floor
    scored = score_offers(test_df, model1, model2, features_m1, features_m2)
    assigned = greedy_allocate(scored["exp_revenue"])
    assigned = enforce_price_floor(assigned, scored["p_convert"], ...)

Caveats:
    - enforce_price_floor and enforce_coverage must be applied after
      greedy_allocate in that order; reversing breaks the constraint logic.
    - score_offers temporarily sets OFFER_RICHNESS_SERVED to each tier value;
      the original column is restored, but pass a copy if mutation is a concern.
"""

import numpy as np
import pandas as pd
from config import OFFERS, MIN_POINTS, PRICE_FLOOR, OFFER_COVERAGE_FLOOR


def score_offers(test_df, model1, model2, feature_cols_m1, feature_cols_m2):
    """Score each session under each offer. Returns (p_convert, exp_points, exp_revenue) dicts."""
    p_convert, exp_points, exp_revenue = {}, {}, {}

    for offer in OFFERS:
        tmp = test_df.copy()
        tmp["OFFER_RICHNESS_SERVED"] = offer
        tmp["OFFER_VS_LAST_PURCHASE"] = np.where(
            tmp["miss_DAYS_SINCE_LAST_PURCHASE_L12M"] == 0,
            offer - tmp["LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed"],
            0,
        )
        p_convert[offer] = model1.predict_proba(tmp[feature_cols_m1])[:, 1]
        exp_points[offer] = model2.predict(tmp[feature_cols_m2])
        price = 0.03 * (1 - offer)
        exp_revenue[offer] = p_convert[offer] * exp_points[offer] * price

    return p_convert, exp_points, exp_revenue


def greedy_allocate(exp_revenue: dict) -> np.ndarray:
    return pd.DataFrame(exp_revenue).idxmax(axis=1).values


def _weighted_avg_price(assigned, p_convert, exp_points):
    p = np.array([p_convert[o][i] for i, o in enumerate(assigned)])
    pts = np.array([exp_points[o][i] for i, o in enumerate(assigned)])
    prc = np.array([0.03 * (1 - o) for o in assigned])
    return (p * pts * prc).sum() / (p * pts).sum()


def enforce_price_floor(assigned, p_convert, exp_points, exp_revenue):
    """Flip lowest-cost 50%→45% sessions until weighted avg price/point >= PRICE_FLOOR."""
    assigned = assigned.copy()

    if _weighted_avg_price(assigned, p_convert, exp_points) >= PRICE_FLOOR:
        return assigned

    sessions_at_50 = np.where(assigned == 0.50)[0]
    revenue_delta = exp_revenue[0.50] - exp_revenue[0.45]
    flip_order = sessions_at_50[np.argsort(revenue_delta[sessions_at_50])]

    for idx in flip_order:
        assigned[idx] = 0.45
        if _weighted_avg_price(assigned, p_convert, exp_points) >= PRICE_FLOOR:
            break

    return assigned


def enforce_coverage(assigned, exp_revenue):
    """Ensure minimum allocation per tier; reassign sessions with smallest revenue gap."""
    assigned = assigned.copy()
    n = len(assigned)

    for offer_str, floor_pct in OFFER_COVERAGE_FLOOR.items():
        offer = float(offer_str)
        current_pct = (assigned == offer).mean()
        if current_pct >= floor_pct:
            continue

        needed = int(np.ceil(floor_pct * n)) - int((assigned == offer).sum())
        candidates = np.where(assigned != offer)[0]
        next_best = np.array([exp_revenue[offer][i] for i in candidates])
        current_rev = np.array([exp_revenue[assigned[i]][i] for i in candidates])
        delta = current_rev - next_best
        flip_idx = candidates[np.argsort(delta)[:needed]]
        assigned[flip_idx] = offer

    return assigned


def compute_strategy_metrics(assigned, p_convert, exp_points):
    """Return E[Rev/session], avg price/point, and share per offer tier."""
    n = len(assigned)
    p = np.array([p_convert[o][i] for i, o in enumerate(assigned)])
    pts = np.array([exp_points[o][i] for i, o in enumerate(assigned)])
    prc = np.array([0.03 * (1 - o) for o in assigned])

    exp_rev_per_session = (p * pts * prc).mean()
    avg_price_per_point = (p * pts * prc).sum() / (p * pts).sum()

    shares = {o: (assigned == o).mean() for o in OFFERS}
    price_floor_ok = avg_price_per_point >= PRICE_FLOOR
    coverage_ok = all(shares[float(o)] >= f for o, f in OFFER_COVERAGE_FLOOR.items())

    return {
        "E[Rev/session]": round(exp_rev_per_session, 6),
        "Avg Price/Point": round(avg_price_per_point, 6),
        "40% share": round(shares[0.40], 4),
        "45% share": round(shares[0.45], 4),
        "50% share": round(shares[0.50], 4),
        "Price floor ✓": price_floor_ok,
        "Coverage ✓": coverage_ok,
    }
