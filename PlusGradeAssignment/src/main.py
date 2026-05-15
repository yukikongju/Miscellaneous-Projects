"""
Offer allocation pipeline — top-level orchestration only.

Usage:
    uv run python src/main.py          # from project root (sys.path shim handles it)

Caveats:
    - Training is delegated to models/train.py; this script reloads fitted
      artifacts from MLflow, so an active MLflow tracking server is required.
    - Results CSV is written to reports/; the directory must exist or be
      created before running.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

OUTPUT_DIR = Path(__file__).parent.parent / "reports"
MLRUNS_DIR = Path(__file__).parent.parent / "mlruns"

# Allow running directly from src/ or from project root via shim
sys.path.insert(0, str(Path(__file__).parent))

from config import FEATURES_MODEL1, FEATURES_MODEL2, TARGET_M1, OFFERS
from data.loader import load_raw, split_by_member, drop_identifiers
from features.engineering import (
    build_features,
    fit_imputer,
    apply_imputer,
    fit_scaler,
    apply_scaler,
    COLS_TO_IMPUTE,
)
from models.models import build_model, BaseModel
from models.train import train_model1, train_model2
from allocation.strategy import (
    score_offers,
    greedy_allocate,
    enforce_price_floor,
    enforce_coverage,
    compute_strategy_metrics,
)


# ── Step 1 + 2: Load, split, engineer features ───────────────────────────────


def prepare_data():
    print("Loading data…")
    df = load_raw()
    print(f"  {len(df):,} rows, {df['MEMBER_KEY'].nunique():,} unique members")

    train_raw, val_raw, test_raw = split_by_member(df)
    print(f"  Train: {len(train_raw):,}  Val: {len(val_raw):,}  Test: {len(test_raw):,}")

    train_fe = build_features(train_raw)
    val_fe = build_features(val_raw)
    test_fe = build_features(test_raw)

    medians = fit_imputer(train_fe, COLS_TO_IMPUTE)
    train_fe = apply_imputer(train_fe, medians)
    val_fe = apply_imputer(val_fe, medians)
    test_fe = apply_imputer(test_fe, medians)

    # Recompute after imputation so OFFER_VS_LAST_PURCHASE uses imputed values
    for df_ in [train_fe, val_fe, test_fe]:
        df_["OFFER_VS_LAST_PURCHASE"] = np.where(
            df_["miss_DAYS_SINCE_LAST_PURCHASE_L12M"] == 0,
            df_["OFFER_RICHNESS_SERVED"]
            - df_["LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed"],
            0,
        )

    train_fe = drop_identifiers(train_fe)
    val_fe = drop_identifiers(val_fe)
    test_fe = drop_identifiers(test_fe)

    scaler = fit_scaler(train_fe)
    train_scaled = apply_scaler(train_fe, scaler)
    val_scaled = apply_scaler(val_fe, scaler)
    test_scaled = apply_scaler(test_fe, scaler)

    return test_raw, train_fe, val_fe, test_fe, train_scaled, val_scaled, test_scaled


# ── MLflow loader ─────────────────────────────────────────────────────────────


def load_model_from_mlflow(run_id: str, artifact_name: str, model_name: str) -> BaseModel:
    """
    Reload a fitted estimator from MLflow and wrap it back in the correct
    BaseModel subclass so predict_proba / predict_log still work.
    """
    estimator = mlflow.sklearn.load_model(f"runs:/{run_id}/{artifact_name}")
    model = build_model({"name": model_name})
    model.estimator = estimator
    return model


# ── Step 4: Offer Allocation ──────────────────────────────────────────────────


def run_allocation(test_fe, model1, features_m1, model2, features_m2):
    print("\n=== Offer Allocation ===")

    p_convert, exp_points, exp_revenue = score_offers(
        test_fe, model1, model2, features_m1, features_m2
    )

    assigned_greedy = greedy_allocate(exp_revenue)
    assigned_floor = enforce_price_floor(assigned_greedy, p_convert, exp_points, exp_revenue)
    assigned_final = enforce_coverage(assigned_floor, exp_revenue)

    n = len(test_fe)
    strategies = {
        "Always-40% baseline": np.full(n, 0.40),
        "Always-45% baseline": np.full(n, 0.45),
        "Always-50% baseline": np.full(n, 0.50),
        "Uniform random": np.random.default_rng(42).choice(OFFERS, size=n),
        "Historical observed": test_fe["OFFER_RICHNESS_SERVED"].values,
        "Model-driven unconstrained": assigned_greedy,
        "Model-driven constrained": assigned_final,
    }

    rows = [
        {"Strategy": label, **compute_strategy_metrics(assigned, p_convert, exp_points)}
        for label, assigned in strategies.items()
    ]
    comparison = pd.DataFrame(rows).set_index("Strategy")
    print("\n" + comparison.to_string())

    print("\nFinal offer distribution:")
    for o in OFFERS:
        print(f"  {o:.0%}: {(assigned_final == o).mean():.1%} of sessions")

    return assigned_final, comparison, p_convert, exp_points


# ── Entrypoint ────────────────────────────────────────────────────────────────


def main():
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())

    # setting the seed
    # rng = np.random.default_rng(seed=42)
    np.random.seed(42)

    test_raw, train_fe, val_fe, test_fe, train_scaled, val_scaled, test_scaled = prepare_data()

    # Step 3A — train classifiers, get best run reference
    run_id1, model_name1, features_m1 = train_model1(
        train_fe,
        val_fe,
        test_fe,
        train_scaled,
        val_scaled,
        test_scaled,
    )
    model1 = load_model_from_mlflow(run_id1, model_name1, model_name1)

    # Step 3B — train regressors, get best run reference
    run_id2, model_name2, features_m2 = train_model2(
        train_fe,
        val_fe,
        test_fe,
        train_scaled,
        val_scaled,
        test_scaled,
    )
    model2 = load_model_from_mlflow(run_id2, model_name2, model_name2)

    # Step 4 — allocate offers
    assigned_final, comparison, p_convert, exp_points = run_allocation(
        test_fe,
        model1,
        features_m1,
        model2,
        features_m2,
    )

    # Save test set with predictions
    # test_out = test_raw[["SESSION_KEY", "MEMBER_KEY", "SESSION_DATE"]].copy().reset_index(drop=True)
    test_out = test_raw.copy().reset_index(drop=True)
    test_out["assigned_offer"] = assigned_final
    test_out["convert_proba"] = np.array([p_convert[o][i] for i, o in enumerate(assigned_final)])
    test_out["points_purchased"] = np.array(
        [exp_points[o][i] for i, o in enumerate(assigned_final)]
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "test_predictions.csv"
    test_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(test_out):,} rows → {out_path}")

    return assigned_final, comparison


if __name__ == "__main__":
    main()
