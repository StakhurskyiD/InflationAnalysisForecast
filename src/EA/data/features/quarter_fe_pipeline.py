#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_quarterly_features.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Створює розширений датасет `quarterly_data_with_features.csv`
із вихідного `quarterly_data.csv`.

• auto-clean column names                           • вилучає службові колонки
• lags / QoQ / YoY / rolling stats (усі числові)    • сезонні sin/cos та dummy-рік
• мінімізована втрата спостережень (rolling min_periods, flex ROW_THRESH)
"""
from __future__ import annotations
import re, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ────────────────────────────── CONFIG ─────────────────────────────────── #
ROOT       = Path(__file__).resolve().parent.parent.parent.parent
RAW_CSV    = ROOT / "research_data" / "processed_data" / "quarterly_data.csv"
OUT_CSV    = ROOT / "research_data" / "processed_data" / "quarterly_data_with_features.csv"

TARGET     = "inflation_index"   # оригінальна колонка-ціль
HORIZON    = 1                   # квартал наперед
LAGS       = (1, 2, 4)           # 1q, 2q, 1y
ROLL_WINS  = (4, 8)              # ≈ 1 рік та 2 роки
ROW_THRESH = 0.50               # мін. частка non-NaN у рядку після генерації

# ───────────────────────── helper-функції ──────────────────────────────── #
def _snake(s: str) -> str:
    """Перевести в snake_case + прибрати небажані символи (укр/лат)."""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s]", "", s, flags=re.U)
    s = re.sub(r"\s+", "_", s.strip(), flags=re.U)
    return s.lower()


def _make_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """quarter_period → DatetimeIndex (1-й день кварталу)"""
    idx = pd.PeriodIndex(df["quarter_period"], freq="Q").to_timestamp()
    df = df.set_index(idx).sort_index()
    df.index.name = "date_q"
    return df


# ────────────────────────────── PIPELINE ───────────────────────────────── #
def build_features() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(RAW_CSV)

    # ---------- 1. LOAD & BASIC CLEAN ------------------------------------ #
    df = pd.read_csv(RAW_CSV, dtype=str)

    # прибираємо службові / debug-колонки (наприклад, _merge)
    df = df.loc[:, ~df.columns.str.contains(r"_merge$", case=False)]

    # стандартизуємо назви колонок
    df.columns = [_snake(c) for c in df.columns]

    # to numeric
    num_cols = df.columns.difference(["quarter_period", "date"])
    df[num_cols] = (df[num_cols]
                    .apply(lambda s: s.str.replace(",", ".", regex=False))
                    .apply(lambda s: s.str.replace(r"[^\d\.\-]", "", regex=True))
                    .apply(pd.to_numeric, errors="coerce"))

    df = _make_time_index(df)
    start_rows = len(df)

    # ---------- 2. TARGET ------------------------------------------------- #
    df[f"{TARGET}_t+{HORIZON}"] = df[TARGET].shift(-HORIZON)

    # ---------- 3. DATE / SEASONAL FEATS --------------------------------- #
    df["quarter"] = df.index.quarter
    df["year"]    = df.index.year
    df["sin_q"]   = np.sin(2 * np.pi * df.quarter / 4)
    df["cos_q"]   = np.cos(2 * np.pi * df.quarter / 4)

    # перелік числових колонок для генерації ознак
    base_num = df.select_dtypes("number").columns.difference([f"{TARGET}_t+{HORIZON}"])

    # ---------- 4. LAGS, QoQ, YoY ---------------------------------------- #
    for lag in LAGS:
        df[[f"{c}_lag{lag}" for c in base_num]] = df[base_num].shift(lag)

    df[[f"{c}_qoq" for c in base_num]] = df[base_num].pct_change().fillna(0)
    df[[f"{c}_yoy" for c in base_num]] = df[base_num].pct_change(4).fillna(0)

    # ---------- 5. ROLLING STATS ----------------------------------------- #
    for w in ROLL_WINS:
        m = (df[base_num]
             .rolling(window=w, min_periods=1)
             .mean()
             .add_suffix(f"_mean{w}"))
        s = (df[base_num]
             .rolling(window=w, min_periods=2)   # std вимагає ≥2
             .std()
             .add_suffix(f"_std{w}"))
        df = pd.concat([df, m, s], axis=1)

    # ---------- 6. INTERACTIONS (приклад) -------------------------------- #
    if {"ppi_index_qoq", "inflation_index_qoq"} <= set(df.columns):
        df["ppi_inf_qoq_inter"] = df["ppi_index_qoq"] * df["inflation_index_qoq"]

    # ---------- 7. CLEAN & SAVE ------------------------------------------ #
    # прибираємо останній квартал без таргету та занадто «порожні» рядки
    df = df.dropna(subset=[f"{TARGET}_t+{HORIZON}"])
    min_non_null = int(ROW_THRESH * df.shape[1])
    df = (df
          .dropna(axis=0, thresh=min_non_null)
          .ffill()
          .dropna())

    kept_rows = len(df)
    print(f"Rows kept after feature gen: {kept_rows}/{start_rows}  "
          f"(-{100*(start_rows-kept_rows)/start_rows:.0f} %)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=True)
    print(f"✓ Feature file saved → {OUT_CSV.resolve()}  (shape = {df.shape})")


# ───────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    build_features()
