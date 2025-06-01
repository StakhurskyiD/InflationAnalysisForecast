"""
prepare_tree.py – end-to-end pipeline (winsorize + RF/XGB + SHAP)
Author: <your-name>
"""

from __future__ import annotations

import inspect
import json
import logging
import warnings
from pathlib import Path
from typing import Literal, Sequence, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from tqdm.auto import tqdm
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

# ────────────────────────── logging ─────────────────────────────────────────
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())


# ─────────────────────── helper-функції ─────────────────────────────────────
def _winsorize_iqr(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Обрізає викиди за правилом 1.5 · IQR (in-place, повертає df для chain)."""
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[c] = df[c].clip(lo, hi)
    return df


def _xgb_gain_importance(model: XGBRegressor, feat_names: list[str]) -> np.ndarray:
    """Gain-based importance, впорядкована як feat_names."""
    gain = model.get_booster().get_score(importance_type="gain")

    # ① нові XGBoost повертають справжні назви ознак
    if set(gain) <= set(feat_names):
        return np.array([gain.get(f, 0.0) for f in feat_names])

    # ② старі – 'f0','f1',…
    return np.array([gain.get(f"f{i}", 0.0) for i in range(len(feat_names))])


# ────────────────────────── main pipeline ───────────────────────────────────
def prepare_tree_data_for_xgb(
    raw_csv: Union[str, Path],
    *,
    target_col: str = "inflation_index_t+1",
    drop_cols: Sequence[str] = ("date", "quarter_period"),
    out_dir: Union[str, Path] = (
        "train_results/tree_ready_preprocessing/rf_xgb_dataset"
    ),
    test_frac: float = 0.20,
    n_splits: int = 5,
    cv_mode: Literal["ts", "expanding"] = "ts",
    rf_params: dict | None = None,
    xgb_params: dict | None = None,
    random_state: int = 42,
    early_stopping_rounds: int = 50,
) -> None:
    """
    Повний цикл: препроцесинг → CV → тренування RF / XGB → SHAP → метрики.
    """
    raw_csv, out_dir = Path(raw_csv), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = out_dir / "dataset_ready"
    ds_dir.mkdir(exist_ok=True)

    # ──────────────── 0. LOAD ──────────────────────────────────────────────
    df = pd.read_csv(raw_csv, parse_dates=["date_q"], index_col="date_q")
    X_full = df.drop(columns=[target_col, *drop_cols])
    y_full = df[target_col]

    # ──────────────── 1. TRAIN / TEST SPLIT ───────────────────────────────
    test_size = int(np.ceil(len(df) * test_frac))
    X_train, X_test = X_full.iloc[:-test_size].copy(), X_full.iloc[-test_size:].copy()
    y_train, y_test = y_full.iloc[:-test_size], y_full.iloc[-test_size:]

    # ──────────────── 2. IMPUTATION + WINSORIZE ───────────────────────────
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

    medians = X_train[num_cols].median()
    modes = X_train[cat_cols].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else "missing"
    )
    # заповнюємо пропуски
    X_train[num_cols] = X_train[num_cols].fillna(medians)
    X_test[num_cols] = X_test[num_cols].fillna(medians)
    X_train[cat_cols] = X_train[cat_cols].fillna(modes)
    X_test[cat_cols] = X_test[cat_cols].fillna(modes)

    # winsorize
    _winsorize_iqr(X_train, num_cols)
    _winsorize_iqr(X_test, num_cols)

    # ──────────────── 3. SAVE RAW DATASETS ────────────────────────────────
    X_train.to_csv(ds_dir / "X_train_raw.csv")
    X_test.to_csv(ds_dir / "X_test_raw.csv")
    y_train.to_csv(ds_dir / "y_train.csv")
    y_test.to_csv(ds_dir / "y_test.csv")

    # ──────────────── 4. PREPROCESSOR ─────────────────────────────────────
    num_pipe = Pipeline([("scale", RobustScaler())])
    cat_pipe = Pipeline(
        [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    prep = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    ).set_output(transform="pandas")

    # ──────────────── 5. МОДЕЛІ ───────────────────────────────────────────
    rf_defaults = dict(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    xgb_defaults = dict(
        n_estimators=2_000,          # верхня межа: зупиниться раніше
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=random_state,
    )

    models = {
        "RF": RandomForestRegressor(**(rf_params or rf_defaults)),
        "XGB": XGBRegressor(**(xgb_params or xgb_defaults)),
    }

    # ──────────────── 6. CROSS-VALIDATION ─────────────────────────────────
    if cv_mode == "expanding":
        # простий expanding window
        horizon = len(X_train) // (n_splits + 1)
        splits = [
            (np.arange(0, k * horizon), np.arange(k * horizon, (k + 1) * horizon))
            for k in range(1, n_splits + 1)
        ]
    else:  # default "ts"
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(X_train))

    metrics: dict[str, dict] = {}

    for name, est in models.items():
        pipe = Pipeline([("prep", prep), ("model", est)])
        mae_f, rmse_f, mape_f = [], [], []

        for fold, (tr_idx, val_idx) in enumerate(tqdm(splits, desc=name), 1):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_val)

            mae_f.append(mean_absolute_error(y_val, pred))

            # підтримуємо старі версії sklearn (<0.22)
            try:
                rmse_val = mean_squared_error(y_val, pred, squared=False)
            except TypeError:
                rmse_val = np.sqrt(mean_squared_error(y_val, pred))
            rmse_f.append(rmse_val)

            mape_f.append(np.mean(np.abs((y_val - pred) / y_val)) * 100)

        metrics[name] = {
            "cv": {
                "MAE": float(np.mean(mae_f)),
                "RMSE": float(np.mean(rmse_f)),
                "MAPE": float(np.mean(mape_f)),
            }
        }
        log.info(
            "%s  CV  MAE %.4f  RMSE %.4f  MAPE %.2f%%",
            name,
            metrics[name]["cv"]["MAE"],
            metrics[name]["cv"]["RMSE"],
            metrics[name]["cv"]["MAPE"],
        )

        # ─────── 7. FINAL TRAIN НА ВСЬОМУ TRAIN ───────────────────────────
        # ─── блок FINAL-TRAIN для XGB замінити повністю
        # 7. FINAL TRAIN ON FULL TRAIN --------------------------------------------
        if name == "XGB":
            prep.fit(X_train)
            X_test_prep = prep.transform(X_test)

            fit_kwargs: dict = {
                "model__eval_set": [(X_test_prep, y_test)],
                "model__verbose": False,
            }

            sig = inspect.signature(XGBRegressor.fit).parameters

            if "early_stopping_rounds" in sig:
                # ≥ 1.7  (full build)  – усе ок
                fit_kwargs["model__early_stopping_rounds"] = early_stopping_rounds

            elif "callbacks" in sig:
                # ≈ 1.3 – 1.6 – працює через callbacks
                fit_kwargs["model__callbacks"] = [
                    EarlyStopping(
                        rounds=early_stopping_rounds,
                        save_best=True,
                        data_name="validation_0",
                        metric_name="rmse",
                    )
                ]

            else:
                # 0.9 – 1.2 – ні того, ні того → тренуємо без ES
                warnings.warn(
                    "XGBoost < 1.3 detected – early-stopping недоступний; "
                    "модель тренується на повній кількості trees",
                    RuntimeWarning,
                )

            pipe.fit(X_train, y_train, **fit_kwargs)

        else:
            pipe.fit(X_train, y_train)

        # ─────── hold-out metrics
        y_pred = pipe.predict(X_test)
        try:
            rmse_test = mean_squared_error(y_test, y_pred, squared=False)
        except TypeError:
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

        metrics[name]["test"] = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(rmse_test),
            "MAPE": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100),
        }

        # ─────── save model
        joblib.dump(pipe, out_dir / f"{name.lower()}_model.pkl")

        # ─────── feature importance
        fi = (
            pipe["model"].feature_importances_
            if name == "RF"
            else _xgb_gain_importance(
                pipe["model"], pipe["prep"].get_feature_names_out()
            )
        )
        (
            pd.Series(fi, index=pipe["prep"].get_feature_names_out())
            .sort_values(ascending=False)
            .to_csv(out_dir / f"{name.lower()}_feat_imp.csv")
        )

        # ─────── SHAP для XGB
        if name == "XGB":
            log.info("Computing SHAP values …")
            explainer = shap.TreeExplainer(pipe["model"])
            X_tr_prep = pipe["prep"].transform(X_train)
            shap_vals = explainer.shap_values(X_tr_prep)

            shap.summary_plot(
                shap_vals,
                X_tr_prep,
                feature_names=pipe["prep"].get_feature_names_out(),
                show=False,
            )
            plt.gcf().set_size_inches(9, 6)
            plt.tight_layout()
            plt.savefig(out_dir / "xgb_shap_summary.png", dpi=300)
            plt.close()

    # ──────────────── 8. SAVE METRICS ─────────────────────────────────────
    with open(out_dir / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

    # pretty print
    log.info("======== FINAL METRICS ========")
    for m, res in metrics.items():
        log.info(
            "%s  CV  MAE %.4f  RMSE %.4f  MAPE %.2f%% | "
            "TEST  MAE %.4f  RMSE %.4f  MAPE %.2f%%",
            m,
            res["cv"]["MAE"],
            res["cv"]["RMSE"],
            res["cv"]["MAPE"],
            res["test"]["MAE"],
            res["test"]["RMSE"],
            res["test"]["MAPE"],
        )
    log.info("✓  Done.  Artifacts saved to %s", out_dir)
