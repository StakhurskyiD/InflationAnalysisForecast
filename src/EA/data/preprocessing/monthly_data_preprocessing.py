from pathlib import Path

import pandas as pd
import numpy as np
import re

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from src.EA.data.mapping.column_mappings import monthly_columns_map


# ---------------------- ФУНКЦІЇ ІМПОРТУ ----------------------

def load_and_map_monthly_data(file_path: str) -> pd.DataFrame:
    """
    Завантажує CSV із щомісячними даними і перейменовує колонки
    згідно зі словником monthly_columns_map.
    """
    df = pd.read_csv(file_path, sep=',', decimal='.')
    df.rename(columns=monthly_columns_map, inplace=True)
    return df


# ---------------------- ФУНКЦІЇ ПРЕПРОЦЕСИНГУ ----------------------

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

    :param df: DataFrame з даними
    :param cols: список імен колонок, для яких потрібно видалити викиди
    :return: DataFrame без викидів
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
    # Якщо є кома і немає крапки, то замінюємо кому на крапку
    if ',' in s and '.' not in s:
        s = s.replace(',', '.')
    # Тепер видаляємо все, що не є цифрою або десятковою крапкою
    s_clean = re.sub(r"[^\d\.]", "", s)
    return s_clean if s_clean != "" else np.nan


def load_and_map_monthly_data(file_path: str) -> pd.DataFrame:
    """
    Завантажує CSV із щомісячними даними і перейменовує колонки
    згідно зі словником monthly_columns_map.
    """
    df = pd.read_csv(file_path, sep=',', decimal='.')  # Читаємо з очікуваним десятковим символом '.'
    df.rename(columns=monthly_columns_map, inplace=True)
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


def seasonal_adjustment(df: pd.DataFrame, date_col: str, value_col: str, period: int = 12,
                        model: str = 'additive') -> pd.DataFrame:
    """
    Застосовує сезонний розклад (seasonal decomposition) до заданої колонки числових даних.
    Додає нову колонку зі сезонно скоригованими даними.

    Для усунення дублікованих дат агрегація проводиться лише для value_col,
    шляхом обчислення середнього значення для кожної дати.

    Перед групуванням значення очищаються (залишаються лише цифри і крапка) та примусово конвертуються в число.

    :param df: DataFrame, що містить стовпець дат.
    :param date_col: Ім'я колонки з датами (тип datetime або рядки, що конвертуються в datetime).
    :param value_col: Ім'я числового стовпця для сезонного розкладу.
    :param period: Сезонна періодичність (12 для щомісячних).
    :param model: Тип моделі розкладу ('additive' або 'multiplicative').
    :return: DataFrame з додатковою колонкою value_col_seasonally_adjusted.
    """
    ts = df.copy()
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts.set_index(date_col, inplace=True)

    # Очищення значень у стовпці value_col: перетворюємо значення в рядки, застосовуємо clean_numeric
    ts[value_col] = ts[value_col].astype(str).apply(clean_numeric)
    # Примусово конвертуємо значення у число
    ts[value_col] = pd.to_numeric(ts[value_col], errors='coerce')

    # Групуємо лише стовпець value_col, обчислюючи середнє для кожної унікальної дати
    ts_value = ts[[value_col]].groupby(ts.index).mean()

    ts_value = ts_value.asfreq('MS')  # встановлюємо частоту – початок місяця
    ts_value.dropna(subset=[value_col], inplace=True)

    # Перевірка: якщо спостережень менше ніж 2 повних цикли (period * 2), не проводимо розклад
    if len(ts_value) < period * 2:
        print(
            f"Попередження: недостатньо спостережень для сезонного розкладу (необхідно {period * 2}, отримано {len(ts_value)}).")
        ts_value[f"{value_col}_seasonally_adjusted"] = ts_value[value_col]
        ts_value.reset_index(inplace=True)
        df_out = pd.merge(df, ts_value[[date_col, f"{value_col}_seasonally_adjusted"]],
                          on=date_col, how='left')
        return df_out

    # Виконуємо сезонний розклад
    result = seasonal_decompose(ts_value[value_col], model=model, period=period, extrapolate_trend='freq')

    # Обчислюємо сезонно скоригований ряд:
    if model == 'additive':
        sa_series = ts_value[value_col] - result.seasonal
    elif model == 'multiplicative':
        sa_series = ts_value[value_col] / result.seasonal
    else:
        raise ValueError("Model must be 'additive' or 'multiplicative'")

    ts_value[f"{value_col}_seasonally_adjusted"] = sa_series
    ts_value.reset_index(inplace=True)

    # Об'єднуємо сезонно скориговані дані з початковим DataFrame
    df_out = pd.merge(df, ts_value[[date_col, f"{value_col}_seasonally_adjusted"]], on=date_col, how='left')
    return df_out


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> (bool, float):
    """
    Перевіряє стаціонарність часової серії за допомогою тесту Augmented Dickey-Fuller.

    :param series: Часова серія (pandas Series).
    :param alpha: Рівень значущості (за замовчуванням 0.05).
    :return: Кортеж (is_stationary, p_value).
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    return (p_value < alpha), p_value


def preprocess_monthly_data():
    # 1. Завантаження місячних даних
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    out_dir = project_root / "research_data" / "import_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly_data_path: Path = out_dir / "imported_monthly_data.csv"
    df_monthly = load_and_map_monthly_data(monthly_data_path)

    # Якщо немає колонки 'year_month', створимо її на основі 'year'
    if 'year_month' not in df_monthly.columns:
        df_monthly['year_month'] = pd.to_datetime(df_monthly['year'], format='%Y')

    # 2. Видалення дублікатів та пропусків
    df_clean = remove_duplicates_and_missing(df_monthly)

    # 3. Видалення викидів для числових колонок (крім 'year')
    numeric_cols = [col for col in df_clean.select_dtypes(include=[np.number]).columns if col != 'year']
    df_clean = remove_outliers(df_clean, numeric_cols)

    # 4. Сезонне коригування для колонки 'inflation_index'
    if 'inflation_index' in df_clean.columns:
        df_clean = seasonal_adjustment(df_clean, date_col='year_month', value_col='inflation_index', period=12, model='additive')

    # 5. Перевірка стаціонарності для часової серії
    # Використовуємо сезонно скоригований стовпець, якщо він існує
    ts = df_clean.copy()
    ts['year_month'] = pd.to_datetime(ts['year_month'])
    ts.set_index('year_month', inplace=True)
    ts = ts[~ts.index.duplicated(keep='first')]
    ts = ts.asfreq('MS')

    # Використовуємо колонку для перевірки: якщо є "inflation_index_seasonally_adjusted", використовуємо її; інакше - "inflation_index"
    col_to_test = "inflation_index_seasonally_adjusted" if "inflation_index_seasonally_adjusted" in ts.columns else "inflation_index"
    if col_to_test in ts.columns:
        is_stationary, p_val = check_stationarity(ts[col_to_test])
        print(f"Серія '{col_to_test}' є стаціонарною: {is_stationary} (p-value = {p_val:.4f})")
    else:
        print(f"Колонка '{col_to_test}' не знайдена для перевірки стаціонарності.")

    # 6. Збереження оброблених даних у файл
    # output_path = "../../../data/output/processed_monthly_data.csv"



    out_dir = project_root / "research_data" / "preprocessed_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path: Path = out_dir / "processed_monthly_data.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"Оброблені місячні дані збережено у файл: {output_path}")