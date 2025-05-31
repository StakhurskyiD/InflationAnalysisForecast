#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge two quarterly datasets on `quarter_period` (all in 'YYYYQn' format).

1) Готовий квартальний CSV, зі стовпцем quarter_period (може бути 'YYYYQn' чи дата).
2) Місячні дані, які ресемплять → квартальні, потім формують quarter_period.

Після злиття:
 - відкидаємо дублі,
 - прибираємо колонки суфіксом '_monthly' (якщо вони дублюють),
 - відкидаємо повністю порожні стовпці й рядки з NaN.
"""
import os, sys, argparse
from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parent.parent.parent.parent

quarterly_input_dir: Path = project_root / "research_data" / "preprocessed_data"
quarterly_input_dir.mkdir(parents=True, exist_ok=True)

transformed_quarterly_input_dir = project_root / "research_data" / "processed_data"
transformed_quarterly_input_dir.mkdir(parents=True, exist_ok=True)

quarterly_data_path: Path = quarterly_input_dir / "processed_quarterly_data.csv"
transformed_quarterly_data_path: Path = transformed_quarterly_input_dir / "converted_from_monthly_quarterly_data.csv"\

final_quarterly_data_path: Path = transformed_quarterly_input_dir / "quarterly_data.csv"





DEFAULT_QTR = quarterly_data_path
DEFAULT_MNTH = transformed_quarterly_data_path
DEFAULT_OUT = final_quarterly_data_path

def _clean_numeric(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r"[\'\"\s%]", "", regex=True)
         .str.replace(",", ".", regex=False)
         .str.replace(r"[^0-9.\-]", "", regex=True)
         .pipe(pd.to_numeric, errors="coerce")
    )

def _to_quarter_str(dts: pd.Series) -> pd.Series:
    return dts.dt.year.astype(str) + "Q" + dts.dt.quarter.astype(str)

def parse_args():
    p = argparse.ArgumentParser("Merge quarterly datasets")
    p.add_argument("--qtr-csv",  default=DEFAULT_QTR,  help="ready quarterly CSV")
    p.add_argument("--mnth-csv", default=DEFAULT_MNTH, help="monthly CSV to resample")
    p.add_argument("--out-csv",  default=DEFAULT_OUT,  help="output merged CSV")
    p.add_argument("--min-months", type=int, default=2,
                   help="min monthly obs per quarter to keep the quarter")
    return p.parse_args([] if len(sys.argv)==1 else None)


def merge_quarterly_data():
    args = parse_args()

    # --- 1) Read both files as strings
    df_q = pd.read_csv(args.qtr_csv, dtype=str)
    df_m = pd.read_csv(args.mnth_csv, dtype=str)

    # strip spaces from column names
    df_q.columns = df_q.columns.str.strip()
    df_m.columns = df_m.columns.str.strip()

    # --- 2) Normalize df_q.quarter_period → 'YYYYQn'
    if "quarter_period" not in df_q.columns:
        # знаходимо будь‑яку колонку з датами, генеруємо quarter_period
        date_col = next(c for c in df_q.columns
                        if pd.to_datetime(df_q[c], errors="coerce").notna().mean() > 0.5)
        df_q[date_col] = pd.to_datetime(df_q[date_col], errors="coerce")
        df_q["quarter_period"] = _to_quarter_str(df_q[date_col])
        df_q.drop(columns=[date_col], inplace=True)

    # якщо quarter_period представлений не у форматі 'YYYYQn', а як дата — конвертуємо
    mask_date = df_q["quarter_period"].str.contains(r"-", na=False)
    if mask_date.any():
        df_q.loc[mask_date, "quarter_period"] = (
            pd.to_datetime(df_q.loc[mask_date, "quarter_period"], errors="coerce")
              .pipe(_to_quarter_str)
        )
    df_q["quarter_period"] = df_q["quarter_period"].str.strip()

    # --- 3) Prepare df_m → квартальні й normalize quarter_period
    if "quarter_period" in df_m.columns:
        # уже квартальні
        df_m_q = df_m.copy()
    else:
        # ресемплим із місячних
        # знаходимо колонку-дату
        date_col = next((c for c in df_m.columns
                         if pd.to_datetime(df_m[c], errors="coerce").notna().mean() > 0.5),
                        None)
        if date_col is None:
            raise KeyError("Не знайдено колонку з датами в місячному файлі")

        # чистимо числові
        df_m[date_col] = pd.to_datetime(df_m[date_col], errors="coerce")
        for c in df_m.columns.difference([date_col]):
            mask = df_m[c].astype(str).str.match(r"^[\d\.\-,\s%]+$")
            if mask.mean() >= .8:
                df_m[c] = _clean_numeric(df_m[c])

        df_m = df_m.set_index(date_col).sort_index()
        def agg_if_enough(x):
            return x.mean() if x.count() >= args.min_months else pd.NA

        df_m_q = df_m.resample("QE-DEC").apply(agg_if_enough).reset_index()
        df_m_q["quarter_period"] = _to_quarter_str(df_m_q[date_col])
        df_m_q.drop(columns=[date_col], inplace=True)

    # і на вході для df_m_q теж нормалізуємо, якщо хтось з якихось причин так і лишив дату
    mask_date_m = df_m_q["quarter_period"].str.contains(r"-", na=False)
    if mask_date_m.any():
        df_m_q.loc[mask_date_m, "quarter_period"] = (
            pd.to_datetime(df_m_q.loc[mask_date_m, "quarter_period"], errors="coerce")
              .pipe(_to_quarter_str)
        )
    df_m_q["quarter_period"] = df_m_q["quarter_period"].str.strip()

    # --- 4) Merge on normalized quarter_period
    merged = pd.merge(
        df_q, df_m_q,
        on="quarter_period", how="inner", suffixes=("", "_monthly")
    )

    # drop duplicate-suffixed cols
    merged = merged.loc[:, ~merged.columns.duplicated()]
    to_drop = [c for c in merged.columns if c.endswith("_monthly")]
    merged.drop(columns=to_drop, inplace=True)

    # --- 5) Clean up empty cols/rows
    merged.dropna(axis=1, how="all", inplace=True)
    merged.dropna(axis=0, how="any", inplace=True)

    # --- 6) Save
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"✔ Merged → {args.out_csv} ({merged.shape[0]}×{merged.shape[1]})")

