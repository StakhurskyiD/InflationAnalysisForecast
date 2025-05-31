import pandas as pd
import numpy as np
from pathlib import Path

###############################################################################
# 0. CONFIG
###############################################################################
project_root = Path(__file__).resolve().parent.parent.parent.parent

quarterly_input_dir = project_root / "research_data" / "processed_data"
quarterly_input_dir.mkdir(parents=True, exist_ok=True)
quarterly_data_path: Path = quarterly_input_dir / "quarterly_data.csv"

quarterly_data_with_features_path: Path = quarterly_input_dir / "quarterly_data_with_features.csv"

RAW_FILE = quarterly_data_path          # ваш CSV
OUT_FILE = quarterly_data_with_features_path
TARGET_COL = "inflation_index"
HORIZON = 1    # квартал наперед

# стовпці‑кандидати
LAG_COLS  = ["inflation_index", "core_inflation_index",
             "cpi_index", "total_gdp_deflator",
             "avg_wage_uah", "ppi_index"]           # + exchange_rate, якщо є
DIFF_COLS = ["total_gdp_deflator", "nominal_gdp_index",
             "avg_wage_uah", "ppi_index"]


def add_features_to_quarterly_data():
    ###############################################################################
    # 1. LOAD & BASIC CLEAN
    ###############################################################################
    df = (pd.read_csv(RAW_FILE)
            .replace(",", ".", regex=True)
            .apply(lambda s: s.str.replace(" ", "") if s.dtype == "object" else s)
    )                       # прибираємо пробіли у числах '11,2'
    num_cols = df.columns.difference(["quarter_period", "date"])
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # перетворимо quarter_period -> DatetimeIndex (перший місяць кварталу)
    df["date_q"] = pd.PeriodIndex(df["quarter_period"], freq="Q").to_timestamp()
    df = df.set_index("date_q").sort_index()

    ###############################################################################
    # 2. TARGET
    ###############################################################################
    df[f"{TARGET_COL}_t+{HORIZON}"] = df[TARGET_COL].shift(-HORIZON)

    ###############################################################################
    # 3. LAG FEATURES
    ###############################################################################
    for col in LAG_COLS:
        for lag in [1, 2, 4]:                   # 1‑, 2‑, 4‑квартальні зсуви
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    ###############################################################################
    # 4. Δ / GROWTH FEATURES
    ###############################################################################
    for col in DIFF_COLS:
        df[f"{col}_qoq"] = df[col].pct_change()             # квартал‑до‑кварталу
        df[f"{col}_yoy"] = df[col].pct_change(4)            # рік‑до‑року
        df[f"{col}_diff"] = df[col].diff()                  # абсолютна різниця

    ###############################################################################
    # 5. ROLLING STATS  (4‑кв. вікно)
    ###############################################################################
    for col in ["inflation_index", "ppi_index", "avg_wage_uah"]:
        df[f"{col}_mean4"] = df[col].rolling(4).mean()
        df[f"{col}_std4"]  = df[col].rolling(4).std()

    ###############################################################################
    # 6. COMPOSITE INDICES
    ###############################################################################
    df["real_wage_index"] = df["avg_wage_uah"] / df["cpi_index"]
    df["gdp_gap"]         = df["nominal_gdp_index"] / df["real_gdp_index"] - 1
    df["inflation_gap"]   = df["inflation_index"] - df["core_inflation_index"]

    ###############################################################################
    # 7. INTERACTION EXAMPLE
    ###############################################################################
    df["ppi_infl_gap_interact"] = df["ppi_index_qoq"] * df["inflation_gap"]

    ###############################################################################
    # 8. FINAL CLEAN & SAVE
    ###############################################################################
    # викидаємо рядки без таргету, заповнюємо решту
    df = df.dropna(subset=[f"{TARGET_COL}_t+{HORIZON}"])
    df = df.fillna(method="ffill").dropna()

    # зберігаємо
    df.to_csv(OUT_FILE, index=True)
    print(f"✓  Feature file saved to {OUT_FILE.resolve()}  (shape = {df.shape})")
