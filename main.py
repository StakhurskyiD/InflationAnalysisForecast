#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py – єдиний entry-point:

① import & preprocess  → quarterly_data_with_features.csv
② train RF / XGB (+ SHAP) – зберігає prep.pkl
③ Hansen (2000) one-threshold оцінка
"""
from __future__ import annotations

import argparse, logging, warnings
from pathlib import Path

# → власні модулі
from src.EA.data.data_import_main import import_and_preprocess_research_data
from src.EA.data.features.quarter_fe_pipeline import build_features
from src.EA.data.features.train_prepare_and_fit import prepare_tree_data_for_xgb
from src.EA.models.hannes_threshold_model import run_hansen


# ─────────────────────────── CLI ───────────────────────────────────────── #
def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Full quarterly-forecast pipeline")
    p.add_argument("--root", default=".", help="корінь проєкту (відносний)")
    p.add_argument("--cv", choices=["ts", "expanding"], default="expanding")
    p.add_argument("--early-stop", type=int, default=40)
    p.add_argument("--row-thresh", type=float, default=0.60,
                   help="частка non-NaN рядка при build_features")
    p.add_argument("--boot", type=int, default=1_000,
                   help="bootstrap replications для Hansen")
    p.add_argument("--th-var", default="num__inflation_gap",
                   help="назва стандартизованої порогової змінної (після prep)")
    return p.parse_args()


# ─────────────────────────── MAIN ──────────────────────────────────────── #
def main() -> None:
    args = _cli()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    log = logging.getLogger("main")

    PROJECT = Path(args.root).resolve()

    # 1. Import → quarterly_data.csv
    log.info("⏳  Import & basic preprocessing …")
    import_and_preprocess_research_data()

    # 2. Feature engineering
    log.info("⏳  Feature engineering …")
    build_features(row_thresh=args.row_thresh)
    log.info("✓  quarterly_data_with_features.csv готовий")

    # 3. Tree-pipeline
    raw_csv = PROJECT / "src" / "research_data" / "processed_data" / \
              "quarterly_data_with_features.csv"
    out_dir = PROJECT / "EA" / "train_results" / "tree_ready_preprocessing"
    out_dir.mkdir(parents=True, exist_ok=True)

    rf_params = dict(n_estimators=300, max_depth=8, random_state=42)
    xgb_params = dict(
        n_estimators=800, learning_rate=0.03, max_depth=3,
        min_child_weight=2, subsample=0.7, colsample_bytree=0.8,
        gamma=0.1, reg_alpha=1, reg_lambda=2,
        objective="reg:squarederror", random_state=42,
    )

    log.info("⏳  Training RF & XGB …")
    prepare_tree_data_for_xgb(
        raw_csv=raw_csv,
        target_col="inflation_index_t+1",
        drop_cols=["date", "quarter_period"],
        out_dir=out_dir,
        test_frac=0.20,
        n_splits=4,
        cv_mode=args.cv,
        early_stopping_rounds=args.early_stop,
        rf_params=rf_params,
        xgb_params=xgb_params,
    )
    log.info("🏁  Tree-pipeline artefacts → %s", out_dir.resolve())

    # 4. Hansen one-threshold
    log.info("⏳  Hansen one-threshold estimation …")
    prep_pkl   = out_dir / "prep.pkl"          # 💡 саме prep.pkl, а не xgb_model.pkl
    hansen_out = PROJECT / "EA" / "hansen_out"
    hansen_out.mkdir(exist_ok=True)

    if not prep_pkl.exists():
        log.error("prep.pkl не знайдено (%s). "
                  "Переконайтесь, що prepare_tree_data_for_xgb зберігає трансформер.",
                  prep_pkl)
    else:
        warnings.filterwarnings("ignore")
        run_hansen(
            raw_csv=raw_csv,
            prep_pkl=prep_pkl,
            target="inflation_index_t+1",
            th_var=args.th_var,        # наприклад  num__inflation_gap
            out_dir=hansen_out,
            boot=args.boot,
        )
        log.info("✓  Hansen artefacts → %s", hansen_out.resolve())

    log.info("🎉  ALL DONE")


# ───────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    main()
