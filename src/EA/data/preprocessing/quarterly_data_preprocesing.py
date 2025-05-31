from pathlib import Path

import pandas as pd
import numpy as np
import re
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from src.EA.data.mapping.column_mappings import quarterly_columns_map


def load_and_map_quarterly_data(file_path: str) -> pd.DataFrame:
    """
    Завантажує CSV із квартальними даними і перейменовує колонки згідно зі словником quarterly_columns_map.
    """
    df = pd.read_csv(file_path, sep=',', decimal='.')
    df.rename(columns=quarterly_columns_map, inplace=True)
    return df


def remove_duplicates_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Видаляє дублікати і рядки з пропущеними значеннями.
    """
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def remove_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Видаляє рядки з викидами для заданих числових колонок за методом IQR.
    """
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean


def clean_numeric(x):
    """
    Очищає вхідне значення, залишаючи лише цифри та десяткову крапку.
    Якщо в рядку є кома як десятковий роздільник (і немає крапки),
    вона замінюється на крапку.
    Якщо результат порожній – повертає np.nan.
    """
    s = str(x).strip()
    if ',' in s and '.' not in s:
        s = s.replace(',', '.')
    s_clean = re.sub(r"[^\d\.]", "", s)
    return s_clean if s_clean != "" else np.nan


def seasonal_adjustment(df: pd.DataFrame, date_col: str, value_col: str, period: int = 4,
                        model: str = 'additive') -> pd.DataFrame:
    """
    Застосовує сезонний розклад до заданої колонки числових даних квартального DataFrame.
    Додає нову колонку зі сезонно скоригованими даними.

    Перед групуванням значення очищаються за допомогою clean_numeric і конвертуються в число.
    Якщо після групування кількість спостережень менша за period*2, повертається попередження і дані без сезонного коригування.

    :param df: DataFrame, що містить стовпець дат.
    :param date_col: Ім'я колонки з датами.
    :param value_col: Ім'я числового стовпця для сезонного розкладу.
    :param period: Сезонна періодичність (4 для квартальних даних).
    :param model: Тип моделі розкладу ('additive' або 'multiplicative').
    :return: DataFrame з додатковою колонкою value_col_seasonally_adjusted.
    """
    ts = df.copy()
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts.set_index(date_col, inplace=True)

    ts[value_col] = ts[value_col].astype(str).apply(clean_numeric)
    ts[value_col] = pd.to_numeric(ts[value_col], errors='coerce')

    # Групуємо лише стовпець value_col для кожної дати, обчислюючи середнє значення
    ts_value = ts[[value_col]].groupby(ts.index).mean()

    # Встановлюємо частоту кварталу.
    # Замість застарілого 'S' використовуйте 's' (припускаємо, що для timestamp все гаразд, проте для квартальних даних частота "QS" більше підходить)
    ts_value = ts_value.asfreq('QS')
    ts_value.dropna(subset=[value_col], inplace=True)

    if len(ts_value) < period * 2:
        print(
            f"Попередження: недостатньо спостережень для сезонного розкладу (необхідно {period * 2}, отримано {len(ts_value)}).")
        ts_value[f"{value_col}_seasonally_adjusted"] = ts_value[value_col]
        ts_value.reset_index(inplace=True)
        df_out = pd.merge(df, ts_value[[date_col, f"{value_col}_seasonally_adjusted"]], on=date_col, how='left')
        return df_out

    result = seasonal_decompose(ts_value[value_col], model=model, period=period, extrapolate_trend='freq')

    if model == 'additive':
        sa_series = ts_value[value_col] - result.seasonal
    elif model == 'multiplicative':
        sa_series = ts_value[value_col] / result.seasonal
    else:
        raise ValueError("Model must be 'additive' or 'multiplicative'")

    ts_value[f"{value_col}_seasonally_adjusted"] = sa_series
    ts_value.reset_index(inplace=True)
    df_out = pd.merge(df, ts_value[[date_col, f"{value_col}_seasonally_adjusted"]], on=date_col, how='left')
    return df_out


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> (bool, float):
    """
    Перевіряє стаціонарність часової серії за допомогою тесту Augmented Dickey-Fuller.

    :param series: Часова серія (pandas Series).
    :param alpha: Рівень значущості (0.05 за замовчуванням).
    :return: Кортеж (is_stationary, p_value). Якщо серія пуста, повертається (False, np.nan).
    """
    s = series.dropna()
    if s.empty:
        print("Попередження: Часова серія порожня. Тест стаціонарності не проводиться.")
        return False, np.nan
    result = adfuller(s)
    p_value = result[1]
    return (p_value < alpha), p_value


def preprocess_quarterly_data():
    # 1. Завантаження квартальних даних
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    out_dir = project_root / "research_data" / "import_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    quarterly_data_path = out_dir / "quarterly_data.csv"
    df_quarterly = load_and_map_quarterly_data(quarterly_data_path)

    # Якщо немає колонки 'date', створимо її на основі 'quarter_period'
    if 'date' not in df_quarterly.columns:
        if 'quarter_period' in df_quarterly.columns:
            df_quarterly['quarter_period'] = df_quarterly['quarter_period'].str.replace(" ", "")
            # Застосовуємо 's' замість 'S'
            df_quarterly['date'] = pd.PeriodIndex(df_quarterly['quarter_period'], freq='Q').to_timestamp('s')
        else:
            raise ValueError("Не знайдено колонки для квартальних дат.")

    # 2. Видалення дублікатів та пропусків
    df_clean = remove_duplicates_and_missing(df_quarterly)

    # 3. Видалення викидів для числових колонок (крім 'date')
    numeric_cols = [col for col in df_clean.select_dtypes(include=[np.number]).columns if col != 'date']
    df_clean = remove_outliers(df_clean, numeric_cols)

    # 4. Сезонне коригування для обраної колонки.
    # Приклад: використаємо колонку "real_gdp_index" (якщо така існує)
    if 'real_gdp_index' in df_clean.columns:
        df_clean = seasonal_adjustment(df_clean, date_col='date', value_col='real_gdp_index', period=4,
                                       model='additive')

    # 5. Перевірка стаціонарності для часової серії
    ts = df_clean.copy()
    ts['date'] = pd.to_datetime(ts['date'])
    ts.set_index('date', inplace=True)
    ts = ts[~ts.index.duplicated(keep='first')]
    ts = ts.asfreq('QS')  # "QS" - початок кварталу

    col_to_test = "real_gdp_index_seasonally_adjusted" if "real_gdp_index_seasonally_adjusted" in ts.columns else "real_gdp_index"
    if col_to_test in ts.columns:
        is_stationary, p_val = check_stationarity(ts[col_to_test])
        print(f"Серія '{col_to_test}' є стаціонарною: {is_stationary} (p-value = {p_val:.4f})")
    else:
        print(f"Колонка '{col_to_test}' не знайдена для перевірки стаціонарності.")

    # 6. Збереження оброблених даних у файл
    output_dir = project_root / "research_data" / "preprocessed_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "processed_quarterly_data.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"Оброблені квартальні дані збережено у файл: {output_path}")