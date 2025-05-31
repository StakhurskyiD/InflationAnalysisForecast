# src/EIA/data/imports/fetch_import_data.py

from dotenv import load_dotenv
from pathlib import Path
import os

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

# 1) Завантажимо .env одразу при імпорті модуля
load_dotenv()

# 2) Приватна функція, що читає ключ і повертає Credentials
spreadsheet_id = os.getenv("SPREADSHEET_ID")
def _get_service_account_creds() -> Credentials:
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    print("Creds path:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    if not creds_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS не встановлено в .env")
    creds_file = Path(creds_path).expanduser().resolve()
    if not creds_file.exists():
        raise FileNotFoundError(f"Не знайдено файл ключа: {creds_file}")
    return Credentials.from_service_account_file(
        filename=str(creds_file),
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
    )

def fetch_sheet_as_df(spreadsheet_id: str, sheet_name: str) -> pd.DataFrame:
    """
    Читаємо дані з Google Sheet через gspread + service account key,
    повертаємо Pandas DataFrame.
    """
    creds = _get_service_account_creds()
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(spreadsheet_id)
    worksheet = sh.worksheet(sheet_name)

    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

def fetch_and_save_research_data():
    MONTHLY = "monthly_data"
    QUARTERLY = "quarterly_data"

    # Визначаємо корінь проєкту динамічно
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    out_dir = project_root / "research_data" / "import_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_m = fetch_sheet_as_df(spreadsheet_id, MONTHLY)
    df_q = fetch_sheet_as_df(spreadsheet_id, QUARTERLY)

    m_path = out_dir / "imported_monthly_data.csv"
    q_path = out_dir / "imported_quarterly_data.csv"

    print(f"Saving {len(df_m)}×{len(df_m.columns)} to {m_path}")
    df_m.to_csv(m_path, index=False, encoding="utf-8-sig")

    print(f"Saving {len(df_q)}×{len(df_q.columns)} to {q_path}")
    df_q.to_csv(q_path, index=False, encoding="utf-8-sig")
# import os
# from pathlib import Path
#
# import pandas as pd
# import gspread
# from google.auth import default
# from dotenv import load_dotenv
#
# # Завантажуємо змінні середовища з .env
# load_dotenv()
#
# # Перевіряємо наявність налаштувань
# creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# spreadsheet_id = os.getenv("SPREADSHEET_ID")
# if not creds_path or not spreadsheet_id:
#     raise RuntimeError("Перевірте, що в .env встановлені GOOGLE_APPLICATION_CREDENTIALS та SPREADSHEET_ID")
#
# print("Using credentials file at:", Path(creds_path).resolve())
#
#
# def fetch_sheet_as_df(spreadsheet_id: str, sheet_name: str) -> pd.DataFrame:
#     """
#     Підключаємось до Google Sheets через gspread + ADC (Application Default Credentials),
#     забираємо весь лист і конвертуємо в DataFrame.
#     """
#     creds, _ = default(scopes=['https://www.googleapis.com/auth/spreadsheets.readonly'])
#     gc = gspread.authorize(creds)
#
#     sh = gc.open_by_key(spreadsheet_id)
#     worksheet = sh.worksheet(sheet_name)
#
#     data = worksheet.get_all_values()
#     df = pd.DataFrame(data[1:], columns=data[0])
#     return df
#
#
# def fetch_and_save_research_data():
#     print('Test')
#     # Імена листів у Google Sheet
#     MONTHLY_SHEET = "monthly_data"
#     QUARTERLY_SHEET = "quarterly_data"
#
#     # 1) Знаходимо корінь проєкту (два рівні підйому від src/main.py → project root)
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent
#
#     # 2) Будуємо шлях до папки import_data
#     out_folder = PROJECT_ROOT / "src" / "research_data" / "import_data"
#     out_folder.mkdir(parents=True, exist_ok=True)
#
#     # 3) Файли для збереження
#     out_monthly = out_folder / "monthly_data.csv"
#     out_quarterly = out_folder / "quarterly_data.csv"
#
#     # 4) Завантажуємо та зберігаємо
#     print(f"Fetching sheet '{MONTHLY_SHEET}' …")
#     df_monthly = fetch_sheet_as_df(spreadsheet_id, MONTHLY_SHEET)
#     print(f"  → got {len(df_monthly)} rows × {len(df_monthly.columns)} cols, saving to {out_monthly}")
#     df_monthly.to_csv(out_monthly, index=False, encoding="utf-8-sig")
#
#     print(f"Fetching sheet '{QUARTERLY_SHEET}' …")
#     df_quarterly = fetch_sheet_as_df(spreadsheet_id, QUARTERLY_SHEET)
#     print(f"  → got {len(df_quarterly)} rows × {len(df_quarterly.columns)} cols, saving to {out_quarterly}")
#     df_quarterly.to_csv(out_quarterly, index=False, encoding="utf-8-sig")
#
#     print("Done.")