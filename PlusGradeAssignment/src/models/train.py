"""
Training routines for Model 1 (conversion classifier) and Model 2 (points regressor).

Usage:
    from models.train import train_model1, train_model2
    run_id, name = train_model1(train_fe, val_fe, test_fe, ...)

Caveats:
    - Both functions log every candidate run to MLflow and return only the
      best run_id; callers reload from MLflow rather than keeping objects
      in memory.
    - train_model2 filters to transacting sessions only (FLAG_TRANSACTION==1)
      before fitting — pass the full feature-engineered splits.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

from config import FEATURES_MODEL1, FEATURES_MODEL2, TARGET_M1, TARGET_M2
from models.models import build_model
from models.tuning import tune
from models.selection import importance_selection
from models.evaluate import (
    evaluate_classifier,
    evaluate_classifier_by_segment,
    evaluate_regressor,
)
from utils.mlflow_utils import log_feature_importances, log_model_artifact

warnings.filterwarnings("ignore")


# ── Shared helpers ────────────────────────────────────────────────────────────


def compute_scale_pos_weight(y: pd.Series) -> float:
    return float((y == 0).sum() / (y == 1).sum())


def is_linear(name: str) -> bool:
    return name in {"logistic_regression", "linear_regression", "ridge", "lasso"}


def _plot_calibration(model, X, y, label: str) -> None:
    prob_true, prob_pred = calibration_curve(y, model.predict_proba(X)[:, 1], n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration — {label}")
    ax.legend()
    fig.tight_layout()
    mlflow.log_figure(fig, f"calibration_{label}.png")
    plt.close(fig)


def _plot_scatter(model, X, y, label: str) -> None:
    preds = model.predict(X)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y, preds, alpha=0.3, s=5)
    mn, mx = min(y.min(), preds.min()), max(y.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "r--")
    ax.set_xlabel("Actual points")
    ax.set_ylabel("Predicted points")
    ax.set_title(f"Predicted vs Actual — {label}")
    fig.tight_layout()
    mlflow.log_figure(fig, f"scatter_{label}.png")
    plt.close(fig)


# ── Model 1 — Conversion Probability ─────────────────────────────────────────


def train_model1(
    train_fe,
    val_fe,
    test_fe,
    train_scaled,
    val_scaled,
    test_scaled,
) -> tuple[str, str, list[str]]:
    """
    Train all classifier candidates. Returns (best_run_id, best_model_name,
    best_features) so the caller can reload the artifact from MLflow.
    """
    y_train = train_fe[TARGET_M1].values
    y_val = val_fe[TARGET_M1].values
    y_test = test_fe[TARGET_M1].values
    spw = compute_scale_pos_weight(train_fe[TARGET_M1])

    configs = [
        {"name": "logistic_regression", "C": 1.0},
        {"name": "naive_bayes"},
        {"name": "random_forest_classifier"},
        {"name": "lgbm_classifier", "scale_pos_weight": spw},
        {"name": "xgb_classifier", "scale_pos_weight": spw},
    ]

    best_run_id, best_model_name, best_features, best_auc = None, None, FEATURES_MODEL1, -1.0

    # Disable autolog globally; enabled selectively inside each per-model run.
    mlflow.autolog(disable=True, silent=True)
    mlflow.set_experiment("offer_allocation_model1")
    with mlflow.start_run(run_name="conversion_models"):
        for cfg in configs:
            name = cfg["name"]
            X_tr = train_scaled[FEATURES_MODEL1] if is_linear(name) else train_fe[FEATURES_MODEL1]
            X_val = val_scaled[FEATURES_MODEL1] if is_linear(name) else val_fe[FEATURES_MODEL1]
            X_te = test_scaled[FEATURES_MODEL1] if is_linear(name) else test_fe[FEATURES_MODEL1]

            print(f"\n[M1] Tuning {name}…")
            best_params = tune(cfg, X_tr, y_train, X_val, y_val, task="classification")
            cfg.update(best_params)

            # Feature-selection fit (autolog off — not a run we want to log).
            _tmp = build_model(cfg)
            _tmp.fit(X_tr, y_train)
            try:
                selected = importance_selection(_tmp, X_tr, threshold=0.01)
            except ValueError:
                selected = FEATURES_MODEL1

            X_tr_fs, X_val_fs, X_te_fs = X_tr[selected], X_val[selected], X_te[selected]

            with mlflow.start_run(run_name=name, nested=True) as run:
                # Enable autolog for this fit only: captures all hyperparameters
                # and per-epoch loss curves (LightGBM / XGBoost) automatically.
                # log_models=False because we log the artifact manually below
                # so it is always stored as an sklearn-compatible pickle.
                mlflow.autolog(log_models=False, log_datasets=False, silent=True)
                model_fs = build_model(cfg)
                model_fs.fit(X_tr_fs, y_train, eval_set=[(X_val_fs, y_val)])
                mlflow.autolog(disable=True, silent=True)  # off again immediately

                # Extras autolog does not cover ─────────────────────────────
                mlflow.log_params({"selected_features": str(selected)})

                train_results = evaluate_classifier(
                    model_fs, X_tr_fs, y_train, label=f"{name}_train"
                )
                val_results = evaluate_classifier(model_fs, X_val_fs, y_val, label=name)
                seg_results = evaluate_classifier_by_segment(
                    model_fs,
                    X_val_fs,
                    y_val,
                    segment=val_fe["FLAG_FIRST_TIME_BUYER"].values,
                    label=name,
                )
                test_results = evaluate_classifier(model_fs, X_te_fs, y_test, label=f"{name}_test")

                mlflow.log_metrics(
                    {
                        "train_auc": train_results["auc"],
                        "val_auc": val_results["auc"],
                        "test_auc": test_results["auc"],
                        "train_log_loss": float(log_loss(y_train, model_fs.predict_proba(X_tr_fs))),
                        "val_log_loss": float(log_loss(y_val, model_fs.predict_proba(X_val_fs))),
                        **seg_results,
                    }
                )
                log_feature_importances(model_fs, selected)
                log_model_artifact(model_fs, name, input_example=X_tr_fs.head())
                _plot_calibration(model_fs, X_val_fs, y_val, name)
                run_id = run.info.run_id

            print(f"  val AUC={val_results['auc']}  test AUC={test_results['auc']}")
            print(f"  subgroup AUC: {seg_results}")

            if val_results["auc"] > best_auc:
                best_auc = val_results["auc"]
                best_run_id = run_id
                best_model_name = name
                best_features = selected

    print(f"\nBest M1: {best_model_name}  val AUC={best_auc}  run_id={best_run_id}")
    return best_run_id, best_model_name, best_features


# ── Model 2 — Expected Points Purchased ──────────────────────────────────────


def train_model2(
    train_fe,
    val_fe,
    test_fe,
    train_scaled,
    val_scaled,
    test_scaled,
) -> tuple[str, str, list[str]]:
    """
    Train all regressor candidates on transacting sessions only.
    Returns (best_run_id, best_model_name, best_features).
    """
    tr_mask = train_fe[TARGET_M1] == 1
    val_mask = val_fe[TARGET_M1] == 1
    te_mask = test_fe[TARGET_M1] == 1

    y_train = train_fe.loc[tr_mask, TARGET_M2].values
    y_val = val_fe.loc[val_mask, TARGET_M2].values
    y_test = test_fe.loc[te_mask, TARGET_M2].values

    configs = [
        {"name": "lasso"},
        {"name": "random_forest_regressor"},
        {"name": "xgb_regressor"},
        {"name": "lgbm_regressor", "objective": "quantile"},
    ]

    best_run_id, best_model_name, best_features, best_rmse = None, None, FEATURES_MODEL2, np.inf

    mlflow.autolog(disable=True, silent=True)
    mlflow.set_experiment("offer_allocation_model2")
    with mlflow.start_run(run_name="points_models"):
        for cfg in configs:
            name = cfg["name"]
            X_tr = train_fe.loc[tr_mask, FEATURES_MODEL2]
            X_val = val_fe.loc[val_mask, FEATURES_MODEL2]
            X_te = test_fe.loc[te_mask, FEATURES_MODEL2]

            print(f"\n[M2] Tuning {name}…")
            best_params = tune(cfg, X_tr, y_train, X_val, y_val, task="regression")
            cfg.update(best_params)

            with mlflow.start_run(run_name=name, nested=True) as run:
                mlflow.autolog(log_models=False, log_datasets=False, silent=True)
                model = build_model(cfg)
                model.fit(X_tr, y_train, eval_set=[(X_val, y_val)])
                mlflow.autolog(disable=True, silent=True)

                mlflow.log_params({"features": str(FEATURES_MODEL2)})

                train_results = evaluate_regressor(model, X_tr, y_train, label=f"{name}_train")
                val_results = evaluate_regressor(model, X_val, y_val, label=name)
                test_results = evaluate_regressor(model, X_te, y_test, label=f"{name}_test")

                mlflow.log_metrics(
                    {
                        "train_rmse": train_results["rmse"],
                        "train_mae": train_results["mae"],
                        "train_r2": train_results["r2"],
                        "val_rmse": val_results["rmse"],
                        "val_mae": val_results["mae"],
                        "val_r2": val_results["r2"],
                        "test_rmse": test_results["rmse"],
                        "test_mae": test_results["mae"],
                        "test_r2": test_results["r2"],
                    }
                )
                log_feature_importances(model, FEATURES_MODEL2)
                log_model_artifact(model, name, input_example=X_tr.head())
                _plot_scatter(model, X_val, y_val, name)
                run_id = run.info.run_id

            print(
                f"  train RMSE={train_results['rmse']}  val RMSE={val_results['rmse']}  test RMSE={test_results['rmse']}"
            )

            if val_results["rmse"] < best_rmse:
                best_rmse = val_results["rmse"]
                best_run_id = run_id
                best_model_name = name
                best_features = FEATURES_MODEL2

    print(f"\nBest M2: {best_model_name}  val RMSE={best_rmse}  run_id={best_run_id}")
    return best_run_id, best_model_name, best_features
