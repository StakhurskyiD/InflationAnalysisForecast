# #
# # """
# # Convert monthly data → quarterly (down‑sampling).
# #
# # • Якщо date‑колонка містить тільки рік (YYYY) — “розгортаємо” його
# #   у послідовні місяці починаючи з січня.
# # • Підтримуються числа з комами, лапками, відсотками тощо.
# # • За замовчуванням агрегація = 'mean', можна вказати 'sum', 'median' або dict колонка→функція.
# # """
# #
# # import os
# # import re
# # import argparse
# # import pandas as pd
# #
# #
# # def _clean_numeric(s: pd.Series) -> pd.Series:
# #     return (
# #         s.astype(str)
# #          .str.replace(r'[\'"\s%]', '', regex=True)
# #          .str.replace(',', '.', regex=False)
# #          .str.replace(r'[^0-9.\-]', '', regex=True)
# #          .pipe(pd.to_numeric, errors='coerce')
# #     )
# #
# #
# # def _parse_dates_and_expand_years(df: pd.DataFrame, date_col: str) -> pd.Series:
# #     """
# #     1) Якщо рядок повністю збігається з \d{4},
# #        “розгортаємо” у послідовні місяці для кожної групи того ж року.
# #     2) Інакше:
# #        YYYY     → YYYY-01-01
# #        YYYY-MM  → YYYY-MM-01
# #        YYYY-MM-DD → 그대로
# #     """
# #     raw = df[date_col].astype(str).str.strip()
# #     # маска “тільки рік”
# #     mask_year = raw.str.fullmatch(r'\d{4}')
# #     dates = pd.Series(pd.NaT, index=df.index)
# #
# #     # 1) розгортаємо рік у місяці
# #     if mask_year.any():
# #         # для кожного року окремо нумеруємо рядки в порядку появи
# #         # і призначаємо місяць = порядковий індекс
# #         grp = raw[mask_year]
# #         seq = grp.groupby(grp).cumcount() + 1  # 1,2,3,...
# #         # якщо більше грипа, ніж 12, місяці повторяться циклічно
# #         months = ((seq - 1) % 12) + 1
# #         dates.loc[mask_year] = pd.to_datetime(
# #             grp.str.cat(months.astype(str).str.zfill(2), sep='-')
# #                .str.cat(['-01']*len(months)), format='%Y-%m-%d', errors='coerce'
# #         )
# #
# #     # 2) решту — стандартний парсинг
# #     rest = raw[~mask_year]
# #     # YYYY → YYYY-01-01
# #     rest = rest.where(~rest.str.fullmatch(r'\d{4}'), rest + '-01-01')
# #     # YYYY-MM → YYYY-MM-01
# #     rest = rest.where(~rest.str.fullmatch(r'\d{4}-\d{2}'), rest + '-01')
# #     dates.loc[~mask_year] = pd.to_datetime(rest, errors='coerce')
# #
# #     return dates
# #
# #
# # def monthly_to_quarterly(
# #     input_csv: str,
# #     output_csv: str,
# #     date_col: str = 'year',
# #     agg: str | dict[str, str] = 'mean',
# #     encoding: str | None = None,
# # ) -> pd.DataFrame:
# #     # 1) Load
# #     df = pd.read_csv(input_csv, sep=',', encoding=encoding)
# #
# #     # 2) Parse & expand dates
# #     df[date_col] = _parse_dates_and_expand_years(df, date_col)
# #     df = df.dropna(subset=[date_col])
# #
# #     # 3) Clean numeric candidates
# #     for col in df.columns.difference([date_col]):
# #         m = df[col].astype(str).str.match(r'^[\d\.\-,\s%]+$')
# #         if m.mean() >= 0.8:
# #             df[col] = _clean_numeric(df[col])
# #
# #     # 4) Down‑sample to quarterly
# #     df = df.set_index(date_col).sort_index()
# #     if isinstance(agg, dict):
# #         df_q = df.resample('Q').agg(agg)
# #     else:
# #         if agg not in {'mean', 'sum', 'median'}:
# #             raise ValueError("agg must be 'mean'|'sum'|'median'|dict")
# #         df_q = getattr(df.resample('Q'), agg)(numeric_only=True)
# #
# #     # 5) Write out
# #     df_q = df_q.reset_index().rename(columns={date_col: 'quarter_period'})
# #     os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
# #     df_q.to_csv(output_csv, index=False)
# #     print(f"✔ Quarterly data written → {output_csv}")
# #     return df_q
# #
# #
# # def _build_cli() -> argparse.ArgumentParser:
# #     p = argparse.ArgumentParser(
# #         description="Monthly → quarterly down‑sampling with year→month expansion"
# #     )
# #     p.add_argument('input_csv', nargs='?',
# #                    default='../output/processed_monthly_data.csv',
# #                    help='Path to monthly CSV')
# #     p.add_argument('output_csv', nargs='?',
# #                    default='../output/processed_quarterly_data_monthly.csv',
# #                    help='Where to write quarterly CSV')
# #     p.add_argument('--date-col', default='year',
# #                    help='Column holding the date')
# #     p.add_argument('--agg', default='mean',
# #                    help="'mean','sum','median' or JSON‑like dict")
# #     p.add_argument('--encoding', default=None,
# #                    help='CSV encoding if not utf‑8')
# #     return p
# #
# #
# # def main() -> None:
# #     import sys, json
# #     parser = _build_cli()
# #     args = parser.parse_args([] if len(sys.argv)==1 else None)
# #
# #     # try parse agg as dict
# #     try:
# #         maybe = json.loads(args.agg)
# #         agg = maybe if isinstance(maybe, dict) else args.agg
# #     except:
# #         agg = args.agg
# #
# #     monthly_to_quarterly(
# #         input_csv   = args.input_csv,
# #         output_csv  = args.output_csv,
# #         date_col    = args.date_col,
# #         agg         = agg,
# #         encoding    = args.encoding
# #     )
# #
# #
# # if __name__ == '__main__':
# #     main()
# #
# #
# #
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Convert monthly data → quarterly (down‑sampling), with:
#  - year-only rows expanded monthly (Jan→Dec cyclically)
#  - support for commas, quotes, %, spaces in numbers
#  - default agg='mean' (also 'sum','median' or dict per-column)
#  - quarter_period in 'YYYYQn' format
#  - drop original date column from output
# """
#
# import os
# import re
# import argparse
# import pandas as pd
#
# def _clean_numeric(s: pd.Series) -> pd.Series:
#     """Очистити рядки типу '186,5 %' → 186.5"""
#     return (
#         s.astype(str)
#          .str.replace(r'[\'"\s%]', '', regex=True)
#          .str.replace(',', '.', regex=False)
#          .str.replace(r'[^0-9.\-]', '', regex=True)
#          .pipe(pd.to_numeric, errors='coerce')
#     )
#
# def _parse_dates_and_expand_years(df: pd.DataFrame, date_col: str) -> pd.Series:
#     """
#     1) Рядки YYYY → розгорнути циклічно по місяцях 1..12
#     2) YYYY-MM → YYYY-MM-01
#     3) YYYY-MM-DD → 그대로
#     """
#     raw = df[date_col].astype(str).str.strip()
#     mask_year = raw.str.fullmatch(r'\d{4}')
#     dates = pd.Series(pd.NaT, index=df.index)
#
#     # 1) Expand pure years
#     if mask_year.any():
#         grp = raw[mask_year]
#         seq = grp.groupby(grp).cumcount() + 1
#         months = ((seq - 1) % 12) + 1
#         y_m = grp + "-" + months.astype(str).str.zfill(2)
#         dates.loc[mask_year] = pd.to_datetime(y_m + "-01", format="%Y-%m-%d", errors='coerce')
#
#     # 2) The rest: add missing day
#     rest = raw[~mask_year]
#     rest = rest.where(~rest.str.fullmatch(r'\d{4}'), rest + "-01-01")
#     rest = rest.where(~rest.str.fullmatch(r'\d{4}-\d{2}'), rest + "-01")
#     dates.loc[~mask_year] = pd.to_datetime(rest, errors='coerce')
#
#     return dates
#
# def monthly_to_quarterly(
#     input_csv: str,
#     output_csv: str,
#     date_col: str = "year_month",
#     agg: str | dict[str,str] = "mean",
#     encoding: str | None = None,
# ) -> pd.DataFrame:
#     # 1) Load
#     df = pd.read_csv(input_csv, sep=",", encoding=encoding)
#
#     # 2) Parse & expand dates
#     df[date_col] = _parse_dates_and_expand_years(df, date_col)
#     df = df.dropna(subset=[date_col])
#
#     # 3) Clean numeric-like columns
#     for col in df.columns.difference([date_col]):
#         # якщо ≥80% рядків «числові» — чистимо
#         mask = df[col].astype(str).str.match(r'^[\d\.\-,\s%]+$')
#         if mask.mean() >= 0.8:
#             df[col] = _clean_numeric(df[col])
#
#     # 4) Resample to quarterly
#     df = df.set_index(date_col).sort_index()
#     if isinstance(agg, dict):
#         df_q = df.resample("Q").agg(agg)
#     else:
#         if agg not in {"mean","sum","median"}:
#             raise ValueError("agg must be 'mean'|'sum'|'median'|dict")
#         df_q = getattr(df.resample("Q"), agg)(numeric_only=True)
#
#     # 5) Build quarter_period and drop original index
#     df_q = df_q.reset_index()
#     # convert Timestamp → 'YYYYQn'
#     df_q["quarter_period"] = df_q[date_col].dt.to_period("Q").astype(str)
#     # drop the date_col entirely
#     df_q.drop(columns=[date_col], inplace=True)
#
#     # 6) Write out
#     os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
#     df_q.to_csv(output_csv, index=False)
#     print(f"✔ Quarterly data written → {output_csv} ({len(df_q)} rows)")
#     return df_q
#
# def _build_cli() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser(
#         description="Monthly → quarterly down‑sampling with year→month expansion"
#     )
#     p.add_argument(
#         "input_csv", nargs="?",
#         default="../output/processed_monthly_data.csv",
#         help="Path to monthly CSV"
#     )
#     p.add_argument(
#         "output_csv", nargs="?",
#         default="../output/processed_quarterly_data_monthly.csv",
#         help="Where to write quarterly CSV"
#     )
#     p.add_argument(
#         "--date-col", default="year_month",
#         help="Column holding the date (YYYY, YYYY-MM or YYYY-MM-DD)"
#     )
#     p.add_argument(
#         "--agg", default="mean",
#         help="'mean','sum','median' or JSON-like dict, e.g. '{\"cpi_index\":\"sum\"}'"
#     )
#     p.add_argument(
#         "--encoding", default=None,
#         help="CSV encoding if not utf‑8 (e.g. cp1251)"
#     )
#     return p
#
# def main() -> None:
#     import sys, json
#     parser = _build_cli()
#     args = parser.parse_args([] if len(sys.argv)==1 else None)
#
#     # parse agg if it's a dict-string
#     try:
#         maybe = json.loads(args.agg)
#         agg = maybe if isinstance(maybe, dict) else args.agg
#     except:
#         agg = args.agg
#
#     monthly_to_quarterly(
#         input_csv  = args.input_csv,
#         output_csv = args.output_csv,
#         date_col   = args.date_col,
#         agg        = agg,
#         encoding   = args.encoding,
#     )
#
# if __name__ == "__main__":
#     main()



"""
Convert monthly data → quarterly (down‑sampling).

• Якщо date‑колонка містить тільки рік (YYYY) — “розгортаємо” його
  у послідовні місяці починаючи з січня.
• Підтримуються числа з комами, лапками, відсотками тощо.
• За замовчуванням агрегація = 'mean', можна вказати 'sum', 'median' або dict колонка→функція.
"""

import os
import re
import argparse
from pathlib import Path

import pandas as pd


def _clean_numeric(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(r'[\'"\s%]', '', regex=True)
         .str.replace(',', '.', regex=False)
         .str.replace(r'[^0-9.\-]', '', regex=True)
         .pipe(pd.to_numeric, errors='coerce')
    )


def _parse_dates_and_expand_years(df: pd.DataFrame, date_col: str) -> pd.Series:
    """
    1) Якщо рядок повністю збігається з \d{4},
       “розгортаємо” у послідовні місяці для кожної групи того ж року.
    2) Інакше:
       YYYY     → YYYY-01-01
       YYYY-MM  → YYYY-MM-01
       YYYY-MM-DD → 그대로
    """
    raw = df[date_col].astype(str).str.strip()
    # маска “тільки рік”
    mask_year = raw.str.fullmatch(r'\d{4}')
    dates = pd.Series(pd.NaT, index=df.index)

    # 1) розгортаємо рік у місяці
    if mask_year.any():
        # для кожного року окремо нумеруємо рядки в порядку появи
        # і призначаємо місяць = порядковий індекс
        grp = raw[mask_year]
        seq = grp.groupby(grp).cumcount() + 1  # 1,2,3,...
        # якщо більше грипа, ніж 12, місяці повторяться циклічно
        months = ((seq - 1) % 12) + 1
        dates.loc[mask_year] = pd.to_datetime(
            grp.str.cat(months.astype(str).str.zfill(2), sep='-')
               .str.cat(['-01']*len(months)), format='%Y-%m-%d', errors='coerce'
        )

    # 2) решту — стандартний парсинг
    rest = raw[~mask_year]
    # YYYY → YYYY-01-01
    rest = rest.where(~rest.str.fullmatch(r'\d{4}'), rest + '-01-01')
    # YYYY-MM → YYYY-MM-01
    rest = rest.where(~rest.str.fullmatch(r'\d{4}-\d{2}'), rest + '-01')
    dates.loc[~mask_year] = pd.to_datetime(rest, errors='coerce')

    return dates


def monthly_to_quarterly(
    input_csv: str,
    output_csv: str,
    date_col: str = 'year',
    agg: str | dict[str, str] = 'mean',
    encoding: str | None = None,
) -> pd.DataFrame:
    # 1) Load
    df = pd.read_csv(input_csv, sep=',', encoding=encoding)

    # 2) Parse & expand dates
    df[date_col] = _parse_dates_and_expand_years(df, date_col)
    df = df.dropna(subset=[date_col])

    # 3) Clean numeric candidates
    for col in df.columns.difference([date_col]):
        m = df[col].astype(str).str.match(r'^[\d\.\-,\s%]+$')
        if m.mean() >= 0.8:
            df[col] = _clean_numeric(df[col])

    # 4) Down‑sample to quarterly
    df = df.set_index(date_col).sort_index()
    if isinstance(agg, dict):
        df_q = df.resample('Q').agg(agg)
    else:
        if agg not in {'mean', 'sum', 'median'}:
            raise ValueError("agg must be 'mean'|'sum'|'median'|dict")
        df_q = getattr(df.resample('Q'), agg)(numeric_only=True)

    # 5) Write out
    df_q = df_q.reset_index().rename(columns={date_col: 'quarter_period'})
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df_q.to_csv(output_csv, index=False)
    print(f"✔ Quarterly data written → {output_csv}")
    return df_q


def _build_cli() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    input_dir = project_root / "research_data" / "preprocessed_data"
    input_dir.mkdir(parents=True, exist_ok=True)

    output_dir = project_root / "research_data" / "processed_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path: Path = input_dir / "processed_monthly_data.csv"
    output_path: Path = output_dir / "converted_from_monthly_quarterly_data.csv"


    monthly_data_path: Path = input_dir / "processed_monthly_data.csv"
    p = argparse.ArgumentParser(
        description="Monthly → quarterly down‑sampling with year→month expansion"
    )
    p.add_argument('input_csv', nargs='?',
                   default=input_path,
                   help='Path to monthly CSV')
    p.add_argument('output_csv', nargs='?',
                   default=output_path,
                   help='Where to write quarterly CSV')
    p.add_argument('--date-col', default='year',
                   help='Column holding the date')
    p.add_argument('--agg', default='mean',
                   help="'mean','sum','median' or JSON‑like dict")
    p.add_argument('--encoding', default=None,
                   help='CSV encoding if not utf‑8')
    return p


def convert_data_from_monthly_to_quarterly_range() -> None:
    import sys, json
    parser = _build_cli()
    args = parser.parse_args([] if len(sys.argv)==1 else None)

    # try parse agg as dict
    try:
        maybe = json.loads(args.agg)
        agg = maybe if isinstance(maybe, dict) else args.agg
    except:
        agg = args.agg

    monthly_to_quarterly(
        input_csv   = args.input_csv,
        output_csv  = args.output_csv,
        date_col    = args.date_col,
        agg         = agg,
        encoding    = args.encoding
    )