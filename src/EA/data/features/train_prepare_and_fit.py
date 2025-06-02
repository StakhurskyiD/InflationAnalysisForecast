#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_tree_data_for_xgb.py
────────────────────────────
End-to-end pipeline: clean → winsorize → train RF & XGB → SHAP →
метрики → зберігає всі артефакти у вибраний каталог.

Артефакти
---------
dataset_ready/
    X_train_raw.csv  X_test_raw.csv  y_train.csv  y_test.csv
rf_model.pkl   xgb_model.pkl
rf_feat_imp.csv  xgb_feat_imp.csv
prep.pkl                 ← 💡 тепер обов’язково зберігається
xgb_shap_summary.png
metrics.json
"""
from __future__ import annotations
import inspect, json, logging, warnings
from pathlib import Path
from typing import Literal, Sequence, Union

import joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from tqdm.auto import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

# ────────────────────────── logging ────────────────────────────────────── #
log = logging.getLogger("prepare_tree")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# ───────────────────── helper-utilities ─────────────────────────────────── #
def _winsorize_iqr(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """In-place 1.5 · IQR clipping."""
    for c in cols:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

def _xgb_gain_importance(model: XGBRegressor, names: list[str]) -> np.ndarray:
    gain = model.get_booster().get_score(importance_type="gain")
    if set(gain) <= set(names):                       # нові XGB
        return np.array([gain.get(f, 0.) for f in names])
    return np.array([gain.get(f"f{i}", 0.) for i in range(len(names))])

# ─────────────────────────── PIPELINE ───────────────────────────────────── #
def prepare_tree_data_for_xgb(
    raw_csv: Union[str, Path],
    *,
    target_col: str = "inflation_index_t+1",
    drop_cols: Sequence[str] = ("date", "quarter_period"),
    out_dir: Union[str, Path] = "train_results/tree_run",
    test_frac: float = .20,
    n_splits: int = 4,
    cv_mode: Literal["ts", "expanding"] = "ts",
    rf_params: dict | None = None,
    xgb_params: dict | None = None,
    random_state: int = 42,
    early_stopping_rounds: int = 40,
) -> None:
    out_dir = Path(out_dir)
    (out_dir / "dataset_ready").mkdir(parents=True, exist_ok=True)
    ds_dir = out_dir / "dataset_ready"

    # 0 ── LOAD ──────────────────────────────────────────────────────────── #
    df = pd.read_csv(raw_csv, parse_dates=["date_q"], index_col="date_q")
    X_full = df.drop(columns=[target_col, *drop_cols])
    y_full = df[target_col]

    # 1 ── TRAIN / TEST split ────────────────────────────────────────────── #
    test_size = max(1, int(np.ceil(len(df) * test_frac)))
    X_train, X_test = X_full.iloc[:-test_size].copy(), X_full.iloc[-test_size:].copy()
    y_train, y_test = y_full.iloc[:-test_size], y_full.iloc[-test_size:]

    # 2 ── IMPUTE + WINSORIZE ────────────────────────────────────────────── #
    num_cols = X_train.select_dtypes(np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

    med = X_train[num_cols].median()
    if cat_cols:
        mode = X_train[cat_cols].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "missing")

    for D in (X_train, X_test):
        D[num_cols] = D[num_cols].fillna(med)
        if cat_cols:
            D[cat_cols] = D[cat_cols].fillna(mode)

    _winsorize_iqr(X_train, num_cols)
    _winsorize_iqr(X_test,  num_cols)

    # save raw splits
    X_train.to_csv(ds_dir / "X_train_raw.csv")
    X_test.to_csv(ds_dir / "X_test_raw.csv")
    y_train.to_csv(ds_dir / "y_train.csv")
    y_test.to_csv(ds_dir / "y_test.csv")

    # 3 ── PREPROCESSOR  (scale + OHE) ───────────────────────────────────── #
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("sc", RobustScaler())]), num_cols),
            ("cat", Pipeline(
                [("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
            ), cat_cols),
        ],
        remainder="drop",
    ).set_output(transform="pandas")

    # 💡  Зберігаємо трансформер одразу (для Hansen-моделі та репро)

    # 4 ── MODELS  + дефолти ─────────────────────────────────────────────── #
    rf_def = dict(n_estimators=300, max_depth=None, max_features="sqrt",
                  min_samples_leaf=5, n_jobs=-1, random_state=random_state)
    xgb_def = dict(
        n_estimators=2_000, learning_rate=.02,
        max_depth=4, min_child_weight=2,
        subsample=.7, colsample_bytree=.8, gamma=.1,
        reg_alpha=1., reg_lambda=2.,
        objective="reg:squarederror",
        n_jobs=-1, random_state=random_state,
    )

    models = {
        "RF": RandomForestRegressor(**(rf_params or rf_def)),
        "XGB": XGBRegressor(**(xgb_params or xgb_def)),
    }

    # 5 ── CV SPLITS ------------------------------------------------------- #
    if cv_mode == "expanding":
        h = len(X_train) // (n_splits + 1)
        splits = [(np.arange(0, k*h), np.arange(k*h, (k+1)*h)) for k in range(1, n_splits+1)]
    else:
        splits = list(TimeSeriesSplit(n_splits=n_splits).split(X_train))

    metrics: dict[str, dict] = {}

    # 6 ── LOOP over models ------------------------------------------------ #
    for name, est in models.items():
        pipe = Pipeline([("prep", prep), ("model", est)])

        cv_mae, cv_rmse, cv_mape = [], [], []
        for tr, vl in tqdm(splits, desc=f"CV-{name}"):
            X_tr, y_tr = X_train.iloc[tr], y_train.iloc[tr]
            X_vl, y_vl = X_train.iloc[vl], y_train.iloc[vl]

            pipe.fit(X_tr, y_tr)
            pr = pipe.predict(X_vl)

            cv_mae.append(mean_absolute_error(y_vl, pr))
            cv_rmse.append(np.sqrt(mean_squared_error(y_vl, pr)))
            cv_mape.append(np.mean(np.abs((y_vl - pr) / y_vl))*100)

        metrics[name] = {"cv": {
            "MAE": float(np.mean(cv_mae)),
            "RMSE": float(np.mean(cv_rmse)),
            "MAPE": float(np.mean(cv_mape)),
        }}
        log.info("%s  CV-MAE %.4f  RMSE %.4f  MAPE %.2f%%",
                 name, *metrics[name]["cv"].values())

        # 7 ── FINAL FIT  (early-stop для XGB) ----------------------------- #
        if name == "XGB":
            prep.fit(X_train); X_test_p = prep.transform(X_test)
            joblib.dump(prep, out_dir / "prep.pkl")

            fit_kw = {"model__eval_set": [(X_test_p, y_test)], "model__verbose": 0}
            if "early_stopping_rounds" in inspect.signature(XGBRegressor.fit).parameters:
                fit_kw["model__early_stopping_rounds"] = early_stopping_rounds
            elif "callbacks" in inspect.signature(XGBRegressor.fit).parameters:
                fit_kw["model__callbacks"] = [EarlyStopping(rounds=early_stopping_rounds, save_best=True)]
            pipe.fit(X_train, y_train, **fit_kw)
        else:
            pipe.fit(X_train, y_train)

        # 8 ── TEST metrics + save model ---------------------------------- #
        y_pred = pipe.predict(X_test)
        metrics[name]["test"] = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAPE": float(np.mean(np.abs((y_test - y_pred) / y_test))*100),
        }
        joblib.dump(pipe, out_dir / f"{name.lower()}_model.pkl")

        # feature importance
        fi = (pipe["model"].feature_importances_
              if name == "RF"
              else _xgb_gain_importance(pipe["model"], pipe["prep"].get_feature_names_out()))
        pd.Series(fi, index=pipe["prep"].get_feature_names_out())\
          .sort_values(ascending=False)\
          .to_csv(out_dir / f"{name.lower()}_feat_imp.csv")

        # SHAP для XGB
        if name == "XGB":
            log.info("…computing SHAP")
            expl = shap.TreeExplainer(pipe["model"])
            sv   = expl.shap_values(prep.transform(X_train))
            shap.summary_plot(
                sv, prep.transform(X_train),
                feature_names=prep.get_feature_names_out(),
                show=False)
            plt.gcf().set_size_inches(9,6); plt.tight_layout()
            plt.savefig(out_dir / "xgb_shap_summary.png", dpi=300); plt.close()

    # 9 ── SAVE metrics ---------------------------------------------------- #
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)

    log.info("======== FINAL METRICS ========")
    for m, res in metrics.items():
        log.info("%s  CV-MAE %.4f | TEST-MAE %.4f  RMSE %.4f  MAPE %.2f%%",
                 m, res['cv']['MAE'], res['test']['MAE'],
                 res['test']['RMSE'], res['test']['MAPE'])
    log.info("✓  Done.  All artifacts → %s", out_dir.resolve())


# ─────────────────────────── CLI entry ─────────────────────────────────── #
if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser("Train RF & XGB on quarterly dataset")
    cli.add_argument("csv", help="src/…/quarterly_data_with_features.csv")
    cli.add_argument("--out-dir", default="train_results/tree_run")
    cli.add_argument("--cv", choices=["ts", "expanding"], default="ts")
    cli.add_argument("--early-stop", type=int, default=40)
    args = cli.parse_args()

    prepare_tree_data_for_xgb(
        args.csv,
        out_dir=args.out_dir,
        cv_mode=args.cv,
        early_stopping_rounds=args.early_stop,
    )
