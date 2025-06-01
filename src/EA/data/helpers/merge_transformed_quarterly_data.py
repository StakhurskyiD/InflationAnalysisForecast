#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_quarterly_data.py
~~~~~~~~~~~~~~~~~~~~~~~
Згортання / злиття квартальних і ре-семплінг-місячних даних у єдиний
квартальний датасет із мінімальними втратами спостережень + докладна
діагностика.

USAGE (CLI)
-----------
python merge_quarterly_data.py \
       --qtr-csv  research_data/preprocessed_data/processed_quarterly_data.csv \
       --mnth-csv research_data/processed_data/converted_from_monthly_quarterly_data.csv \
       --out-csv  research_data/processed_data/quarterly_data.csv \
       --join outer            # outer|left|inner
       --min-months 1          # скільки місяців потрібно для кварталу
       --row-thresh 0.6        # min частка non-NA в рядку (0–1)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# ───────────────────────────── конфіг логера ───────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ───────────────────────────── util-функції ────────────────────────────── #
def _clean_numeric(s: pd.Series) -> pd.Series:
    """Грубе «очищення» числових колонок, де трапляються пробіли/коми/%."""
    return (
        s.astype(str)
         .str.replace(r"[\'\"\s%]", "", regex=True)
         .str.replace(",", ".", regex=False)
         .str.replace(r"[^0-9.\-]", "", regex=True)
         .pipe(pd.to_numeric, errors="coerce")
    )


def _to_quarter_str(dts: pd.Series) -> pd.Series:
    """Timestamp → 'YYYYQn'."""
    return dts.dt.year.astype(str) + "Q" + dts.dt.quarter.astype(str)


def _normalize_qp(series: pd.Series) -> pd.Series:
    """Уніфікуємо quarter_period до 'YYYYQn' + прибираємо пробіли."""
    qp_dates = pd.to_datetime(series, errors="coerce")
    mask_dt = qp_dates.notna()
    series = series.copy()
    if mask_dt.any():
        series.loc[mask_dt] = _to_quarter_str(qp_dates[mask_dt])
    return series.str.strip()


# ───────────────────────────── аргументи CLI ───────────────────────────── #
def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]          # корінь репо
    default_q = root / "research_data" / "preprocessed_data" / "processed_quarterly_data.csv"
    default_m = root / "research_data" / "processed_data" / "converted_from_monthly_quarterly_data.csv"
    default_o = root / "research_data" / "processed_data" / "quarterly_data.csv"

    p = argparse.ArgumentParser("Merge quarterly + resampled-monthly datasets")
    p.add_argument("--qtr-csv",  type=Path, default=default_q, help="already-quarterly CSV")
    p.add_argument("--mnth-csv", type=Path, default=default_m, help="raw monthly CSV")
    p.add_argument("--out-csv",  type=Path, default=default_o, help="output merged CSV")
    p.add_argument("--join", choices=("outer", "left", "inner"), default="left",
                   help="тип злиття; outer ➜ мінімум втрат, left ➜ лише df_q")
    p.add_argument("--min-months", type=int, default=2,
                   help="мінімум місяців, щоб розрахувати середнє по кварталу")
    p.add_argument("--row-thresh", type=float, default=0.6,
                   help="мін. частка non-NA у рядку (0–1) після merge")
    return p.parse_args([] if len(sys.argv) == 1 else None)


# ───────────────────────────── основна логіка ──────────────────────────── #
def merge_quarterly_data() -> None:
    args = _parse_args()
    if not args.qtr_csv.exists() or not args.mnth_csv.exists():
        raise FileNotFoundError("Перевірте шляхи до вхідних файлів")

    # 1. LOAD --------------------------------------------------------------- #
    df_q = pd.read_csv(args.qtr_csv, dtype=str)
    df_m = pd.read_csv(args.mnth_csv, dtype=str)

    log.info("Rows  quarterly: %d | monthly: %d", len(df_q), len(df_m))

    # 2. NORMALISE квартального DF ----------------------------------------- #
    df_q.columns = df_q.columns.str.strip()
    if "quarter_period" not in df_q.columns:
        date_col = next(
            c for c in df_q.columns
            if pd.to_datetime(df_q[c], errors="coerce").notna().mean() > .5
        )
        df_q[date_col] = pd.to_datetime(df_q[date_col], errors="coerce")
        df_q["quarter_period"] = _to_quarter_str(df_q[date_col])
        df_q.drop(columns=[date_col], inplace=True)

    df_q["quarter_period"] = _normalize_qp(df_q["quarter_period"])
    df_q["src_quarter"] = 1

    # 3. MONTHLY → QUARTERLY ------------------------------------------------ #
    if "quarter_period" in df_m.columns:
        df_m_q = df_m.copy()
    else:
        date_col = next(
            (c for c in df_m.columns
             if pd.to_datetime(df_m[c], errors="coerce").notna().mean() > .5),
            None
        )
        if date_col is None:
            raise KeyError("Не знайдено колонку-дату в місячному файлі")

        df_m[date_col] = pd.to_datetime(df_m[date_col], errors="coerce")

        # «чистимо» потенційні числові колонки
        for c in df_m.columns.difference([date_col]):
            if df_m[c].astype(str).str.match(r"^[\d\.\-,\s%]+$").mean() >= .8:
                df_m[c] = _clean_numeric(df_m[c])

        def agg_if_enough(x):
            return x.mean() if x.count() >= args.min_months else pd.NA

        df_m_q = (df_m
                  .set_index(date_col).sort_index()
                  .resample("QE-DEC")
                  .apply(agg_if_enough)
                  .reset_index())

        df_m_q["quarter_period"] = _to_quarter_str(df_m_q[date_col])
        df_m_q.drop(columns=[date_col], inplace=True)

    df_m_q["quarter_period"] = _normalize_qp(df_m_q["quarter_period"])
    df_m_q["src_monthly"] = 1

    # 4. MERGE -------------------------------------------------------------- #
    merged = (
        df_q.merge(df_m_q, on="quarter_period", how=args.join, suffixes=("", "_monthly"),
                   indicator=True)
    )
    log.info("Rows after %s-join: %d", args.join, len(merged))

    # 5. REMOVE duplicate '_monthly' cols ---------------------------------- #
    dup_cols = [c for c in merged.columns if c.endswith("_monthly")]
    merged = merged.drop(columns=dup_cols)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # 6. DROP all-NaN cols & sparse rows ----------------------------------- #
    merged.dropna(axis=1, how="all", inplace=True)
    row_thresh = int(args.row_thresh * merged.shape[1])
    before_rows = len(merged)
    merged.dropna(axis=0, thresh=row_thresh, inplace=True)
    log.info("Dropped %d sparse rows (row_thresh=%.0f%%)",
             before_rows - len(merged), args.row_thresh * 100)

    # 7. SAVE --------------------------------------------------------------- #
    os.makedirs(args.out_csv.parent, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    log.info("✔ Saved merged file → %s  (shape = %s)", args.out_csv, merged.shape)

    # 8. SUMMARY ------------------------------------------------------------ #
    summary = pd.DataFrame({
        "Stage": ["quarterly in", "monthly→Q in", f"after {args.join}", "after drop-rows"],
        "Rows":  [len(df_q),    len(df_m_q),      before_rows,          len(merged)],
    })
    only_m = merged["_merge"].eq("right_only").sum()    # outer-merge case
    only_q = merged["_merge"].eq("left_only").sum()
    both   = merged["_merge"].eq("both").sum()

    print("\n===== MERGE SUMMARY =====")
    print(summary.to_string(index=False))
    if args.join == "outer":
        print(f"\n• кварталів лише з monthly:   {only_m}")
        print(f"• лише з quarterly:           {only_q}")
        print(f"• у двох джерелах:            {both}")


# ─────────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    merge_quarterly_data()
