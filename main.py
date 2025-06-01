# This is a sample Python script.
from src.EA.data.data_import_main import import_and_preprocess_research_data
from src.EA.data.features.train_prepare_and_fit import prepare_tree_data_for_xgb
from src.EA.models.train_rf_xgb_raw import train_models_from_prep


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     import_and_preprocess_research_data()
#     prepare_tree_data_for_xgb()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# імпорт необхідних бібліотек
import logging
from pathlib import Path

# імпорт власних функцій (залежить від структури проекту)

def main() -> None:
    # -------- Логування ----------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    logger = logging.getLogger(__name__)

    # -------- 1. Імпорт та FE ----------------------------------------------
    logger.info("⏳  Import & feature engineering …")
    import_and_preprocess_research_data()          # тут уже створюється .csv
    logger.info("✓  Features готові")

    # -------- 2. Шляхи та параметри ---------------------------------------
    # project_root = коренева папка репозиторію (там, де лежить main.py)
    project_root = Path(__file__).resolve().parent

    raw_csv = project_root / "src" / "research_data" / "processed_data" / \
              "quarterly_data_with_features.csv"

    if not raw_csv.exists():
        raise FileNotFoundError(f"Файл не знайдено: {raw_csv}")

    out_dir = project_root / "EA" / "train_results" / \
              "tree_ready_preprocessing" / "rf_xgb_dataset"

    # гіперпараметри (можна залишити None, щоб узяти дефолтні у prepare_tree_data_for_xgb)
    rf_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
    }
    # xgb_params = {
    #     "n_estimators": 100,
    #     "learning_rate": 0.1,
    #     "max_depth": 6,
    #     "random_state": 42,
    # }

    xgb_params = dict(
        n_estimators=400,  # early-stopping зупинить раніше
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.4,
        min_child_weight=2,
        reg_alpha=2.0,
        reg_lambda=3.0,
        objective="reg:squarederror",
        random_state=42,
    )

    # -------- 3. Тренування -------------------------------------------------
    logger.info("⏳  Training RF & XGB …")
    prepare_tree_data_for_xgb(
        raw_csv=raw_csv,
        target_col="inflation_index_t+1",
        drop_cols=["date", "quarter_period"],
        out_dir=out_dir,
        test_frac=0.20,
        n_splits=3,
        rf_params=rf_params,
        xgb_params=xgb_params,
    )
    logger.info("🏁  Готово. Моделі та метрики збережено у %s", out_dir.resolve())

    # logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_models_from_prep(
        raw_csv="src/research_data/processed_data/quarterly_data_with_features.csv",
        prep_file="EA/train_results/tree_ready_preprocessing/rf_xgb_dataset/xgb_model.pkl",
        target_col="inflation_index_t+1",
        drop_cols=["date", "quarter_period"],
        out_dir="models",
    )
if __name__ == "__main__":
    main()
