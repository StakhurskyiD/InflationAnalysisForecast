# src/pipelines/train_models_from_prep.py
# -----------------------------------------------------------
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Literal, Sequence, Union
import inspect

import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, callback

try:
    from tqdm.auto import tqdm
except ImportError:                   # fallback, якщо tqdm не встановлений
    def tqdm(it, **kw): return it      # type: ignore


def _es_callback(rounds: int = 50):
    """Колбек «рання зупинка» сумісний із XGBoost ≥ 2.0."""
    return callback.EarlyStopping(rounds=rounds, save_best=True)


# ─────────────────────────────────────────────────────────────
def train_models_from_prep(
    raw_csv: Union[str, Path],
    prep_file: Union[str, Path],
    *,
    target_col: str,
    drop_cols: Sequence[str],
    out_dir: Union[str, Path] = "models",
    test_frac: float = 0.20,
    n_splits: int = 3,
    cv_mode: Literal["ts", "expanding"] = "expanding",
    eval_frac: float = 0.50,               # високий %, щоб завжди ≥1 рядок
    rf_params: dict | None = None,
    xgb_params: dict | None = None,
    random_state: int = 42,
    logger: logging.Logger | None = None,
) -> None:
    log = logger or logging.getLogger(__name__)
    raw_csv, prep_file, out_dir = Path(raw_csv), Path(prep_file), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0. DATA & PREP --------------------------------------------------------
    df = pd.read_csv(raw_csv, parse_dates=["date_q"], index_col="date_q")
    X_raw = df.drop(columns=[target_col, *drop_cols])
    y = df[target_col]

    prep: ColumnTransformer = joblib.load(prep_file)
    log.info("✓  prep.pkl завантажено (%d transformers)", len(prep.transformers))

    # 1. HOLD-OUT -----------------------------------------------------------
    test_size = max(1, int(np.ceil(len(df) * test_frac)))
    X_train, X_test = X_raw.iloc[:-test_size], X_raw.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # 2. MODELS -------------------------------------------------------------
    rf_defaults = dict(
        n_estimators=600, max_depth=None, max_features="sqrt",
        min_samples_leaf=10, n_jobs=-1, random_state=random_state
    )
    xgb_defaults = dict(
        n_estimators=2_000, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.5, reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1, random_state=random_state
    )
    models = {
        "RF": RandomForestRegressor(**(rf_params or rf_defaults)),
        "XGB": XGBRegressor(**(xgb_params or xgb_defaults)),
    }

    # 3. CV SPLITS ----------------------------------------------------------
    if cv_mode == "ts":
        splits = list(TimeSeriesSplit(n_splits=n_splits).split(X_train))
    else:  # expanding
        horizon = max(1, len(X_train) // (n_splits + 1))
        splits = [
            (np.arange(0, k * horizon),
             np.arange(k * horizon, min(len(X_train), (k + 1) * horizon)))
            for k in range(1, n_splits + 1)
        ]

    metrics = {"cv_mae": {}, "test_mae": {}}

    for name, est in models.items():
        pipe = Pipeline([("prep", prep), ("model", est)])
        fold_mae: list[float] = []

        for k, (tr_idx, val_idx) in enumerate(tqdm(splits, desc=name), 1):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            if name == "XGB":
                eval_size = max(1, int(len(X_tr) * eval_frac))
                if eval_size < len(X_tr):
                    X_tr_es, y_tr_es = X_tr.iloc[:-eval_size], y_tr.iloc[:-eval_size]
                    X_eval, y_eval = X_tr.iloc[-eval_size:], y_tr.iloc[-eval_size:]
                    _fit_xgb(pipe, X_tr_es, y_tr_es, X_eval, y_eval, rounds=50)
                else:
                    _fit_xgb(pipe, X_tr, y_tr, rounds=50)  # замало рядків → без eval_set
            else:
                pipe.fit(X_tr, y_tr)

            fold_mae.append(mean_absolute_error(y_val, pipe.predict(X_val)))
            log.info("%s  fold %d/%d  MAE=%.3f", name, k, n_splits, fold_mae[-1])

        metrics["cv_mae"][name] = float(np.mean(fold_mae))
        log.info("%s  mean CV-MAE: %.3f", name, metrics["cv_mae"][name])

        # 4. TRAIN ON FULL ---------------------------------------------------
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, out_dir / f"{name.lower()}_model.pkl")

        y_pred_test = pipe.predict(X_test)
        metrics["test_mae"][name] = mean_absolute_error(y_test, y_pred_test)
        pd.Series(y_pred_test, index=X_test.index, name="y_pred") \
          .to_csv(out_dir / f"{name.lower()}_y_test_pred.csv")

        # 5. FEATURE IMPORTANCE ---------------------------------------------
        if name == "RF":
            imp = xgb_gain_importance(pipe[-1], pipe[0].get_feature_names_out())
        else:
            fmap = pipe[-1].get_booster().get_score(importance_type="gain")
            imp = np.array([fmap.get(f, 0.0) for f in pipe[0].get_feature_names_out()])
        pd.Series(imp, index=pipe[0].get_feature_names_out()) \
          .sort_values(ascending=False) \
          .to_csv(out_dir / f"{name.lower()}_feat_imp.csv")

        if "XGB" in name:
            _fit_xgb(pipe, X_train, y_train, rounds=50)  # без eval-set
        else:
            pipe.fit(X_train, y_train)

    # 6. SAVE METRICS -------------------------------------------------------
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("🏁  Готово.  Metрики: %s", metrics)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    train_models_from_prep(
        raw_csv="src/research_data/processed_data/quarterly_data_with_features.csv",
        prep_file="src/research_data/processed_data/prep.pkl",
        target_col="inflation_index_t+1",
        drop_cols=["date", "quarter_period"],
        out_dir="models",
        n_splits=3,
        cv_mode="expanding",
        eval_frac=0.5,
    )


def _has_param(func, name: str) -> bool:
    """Перевіряє, чи ф-ція/метод приймає параметр."""
    return name in inspect.signature(func).parameters


def _fit_xgb(pipe: Pipeline,
             X_tr, y_tr,
             X_eval=None, y_eval=None,
             rounds: int = 50) -> None:
    """Сумісний fit з ранньою зупинкою для всіх версій XGBoost."""
    fit_params = {}
    est_fit = pipe[-1].fit

    if X_eval is not None and y_eval is not None:
        # 1) Нова схема (callbacks)
        if _has_param(est_fit, "callbacks"):
            fit_params["callbacks"] = [callback.EarlyStopping(rounds=rounds, save_best=True)]
            fit_params["eval_set"] = [(X_eval, y_eval)]

        # 2) Стара схема (early_stopping_rounds)
        elif _has_param(est_fit, "early_stopping_rounds"):
            fit_params["early_stopping_rounds"] = rounds
            fit_params["eval_set"] = [(X_eval, y_eval)]

        # 3) Жоден із параметрів не підтримується  → без ES
    pipe.fit(X_tr, y_tr, **fit_params)


def xgb_gain_importance(est, feature_names):
    # витягаємо map fX → ім'я
    fmap = dict(zip(est.feature_names_in_, feature_names))
    raw = est.get_booster().get_score(importance_type="gain")
    return np.array([raw.get(k, 0.0) for k in est.feature_names_in_])