#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_quarterly_features.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Створює розширений датасет `quarterly_data_with_features.csv`
із вихідного `quarterly_data.csv`.

• auto-clean column names                         • вилучає службові колонки
• lags / QoQ / YoY / rolling stats (усі numeric)  • сезонні sin/cos та dummy-рік
• мінімізована втрата спостережень (rolling min_periods, flex ROW_THRESH)
"""
from __future__ import annotations
import argparse, re, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ────────────────────────────── CONFIG ────────────────────────────────── #
ROOT     = Path(__file__).resolve().parents[3]
RAW_CSV  = ROOT / "research_data" / "processed_data" / "quarterly_data.csv"
OUT_CSV  = ROOT / "research_data" / "processed_data" / "quarterly_data_with_features.csv"

TARGET   = "inflation_index"                       # базовий таргет
HORIZON  = 1                                       # квартал наперед
LAGS     = (1, 2, 3, 4, 6, 8)                      # 1q … 2y
ROLL_W   = (4, 8, 12)                              # 1y, 2y, 3y   (mean/std)
# ────────────────────────────── CLI ───────────────────────────────────── #
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Quarterly FE generator")
    p.add_argument("--row-thresh", type=float, default=0.50,
                   help="мін. частка non-NaN у рядку (0-1) після FE (def=0.5)")
    return p.parse_args()

# ───────────────────────── helper-функції ─────────────────────────────── #
def _snake(s: str) -> str:
    """Привести рядок до snake_case + прибрати небажані символи."""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s]", "", s, flags=re.U)
    s = re.sub(r"\s+", "_", s.strip(), flags=re.U)
    return s.lower()

def _make_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """quarter_period  -> DatetimeIndex (перший день кварталу)."""
    idx = pd.PeriodIndex(df["quarter_period"], freq="Q").to_timestamp()
    df = df.set_index(idx).sort_index()
    df.index.name = "date_q"
    return df

# ────────────────────────────── PIPELINE ──────────────────────────────── #
def build_features(row_thresh: float) -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"RAW file not found: {RAW_CSV}")

    # 1. ───── LOAD & basic tidy ─────────────────────────────────────────── #
    df = pd.read_csv(RAW_CSV, dtype=str)

    # службові (_merge etc.)
    df = df.loc[:, ~df.columns.str.contains(r"_merge$", case=False)]

    df.columns = [_snake(c) for c in df.columns]

    # numeric to float
    num_cols = df.columns.difference(["quarter_period", "date"])
    df[num_cols] = (df[num_cols]
                    .apply(lambda s: s.str.replace(",", ".", regex=False))
                    .apply(lambda s: s.str.replace(r"[^\d\.\-]", "", regex=True))
                    .apply(pd.to_numeric, errors="coerce"))

    df = _make_dt_index(df)
    n_start = len(df)

    # 2. ───── target  y_{t+1} ──────────────────────────────────────────── #
    df[f"{TARGET}_t+{HORIZON}"] = df[TARGET].shift(-HORIZON)

    # 3. ───── календарні / сезонні ─────────────────────────────────────── #
    df["quarter"] = df.index.quarter
    df["year"]    = df.index.year
    df["sin_q"]   = np.sin(2 * np.pi * df.quarter / 4)
    df["cos_q"]   = np.cos(2 * np.pi * df.quarter / 4)

    base_num = df.select_dtypes("number").columns.difference([f"{TARGET}_t+{HORIZON}"])

    # 4. ───── лаги, QoQ, YoY ───────────────────────────────────────────── #
    for lag in LAGS:
        df[[f"{c}_lag{lag}" for c in base_num]] = df[base_num].shift(lag)

    df[[f"{c}_qoq" for c in base_num]] = df[base_num].pct_change().fillna(0)
    df[[f"{c}_yoy" for c in base_num]] = df[base_num].pct_change(4).fillna(0)


    # 5. ───── rolling mean / std ───────────────────────────────────────── #
    for w in ROLL_W:
        df = pd.concat(
            [
                df,
                df[base_num]
                  .rolling(w, min_periods=1)
                  .mean()
                  .add_suffix(f"_mean{w}"),
                df[base_num]
                  .rolling(w, min_periods=2)
                  .std()
                  .add_suffix(f"_std{w}"),
            ],
            axis=1,
        )

    df["inflation_gap"] = df["inflation_index"] - df["core_inflation_index"]

    # 6. ───── interaction demo ─────────────────────────────────────────── #
    if {"ppi_index_qoq", "inflation_index_qoq"} <= set(df.columns):
        df["ppi_inf_qoq_inter"] = df["ppi_index_qoq"] * df["inflation_index_qoq"]

    # 7. ───── final clean ──────────────────────────────────────────────── #
    df = df.dropna(subset=[f"{TARGET}_t+{HORIZON}"])
    min_non_null = int(row_thresh * df.shape[1])
    df = (df
          .dropna(axis=0, thresh=min_non_null)   # гнучкий thresh
          .ffill()
          .dropna())

    n_kept = len(df)
    print(f"Rows kept after FE: {n_kept}/{n_start} "
          f"({100*(n_start-n_kept)/n_start:.1f}% dropped) | "
          f"ROW_THRESH={row_thresh}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=True)
    print(f"✓  Features written → {OUT_CSV}   shape={df.shape}")

# ───────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    args = cli()
    build_features(row_thresh=args.row_thresh)
