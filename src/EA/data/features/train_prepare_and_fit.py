"""
Full tree‑ready preprocessing + baseline RF / XGB training.
-----------------------------------------------------------------
INPUT : data/processed_quarter_features.csv   (після Feature Eng.)
OUTPUT:
    dataset_ready/
        X_train.csv, X_test.csv, y_train.csv, y_test.csv
        prep.pkl                 (ColumnTransformer)
    models/
        rf_model.pkl, xgb_model.pkl
        rf_feat_imp.csv, xgb_feat_imp.csv
        cv_mae.json              (5‑fold TS MAE)
"""
import pandas as pd, numpy as np, json, joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def prepare_tree_data_for_xgb():
    # ------------------------------------------------------------------ CONFIG --
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    quarterly_input_dir = project_root / "research_data" / "processed_data"
    quarterly_input_dir.mkdir(parents=True, exist_ok=True)

    quarterly_data_with_features_path: Path = quarterly_input_dir / "quarterly_data_with_features.csv"
    RAW_FE = quarterly_data_with_features_path
    TARGET   = "inflation_index_t+1"
    TEST_FRAC = 0.20               # 20 % останніх кварталів = валідаційний «хвіст»
    N_SPLITS = 5                   # для внутрішнього CV
    # --------------------------------------------------------------------------- #
    output_data_dir = project_root / "EA" / "train_results" / "tree_ready_preprocessing" / "rf_xgb_dataset"

    out_data = output_data_dir;  out_data.mkdir(exist_ok=True)
    out_model = output_data_dir;         out_model.mkdir(exist_ok=True)

    # ------------------------------------------------------------- 0. LOAD ------
    df = pd.read_csv(RAW_FE, parse_dates=['date_q'], index_col='date_q')
    drop_cols = ["date", "quarter_period"]  # <- якщо потрібні, закоментуйте
    X = df.drop(columns=[TARGET] + drop_cols)
    y = df[TARGET]

    # ------------------------------------------------------------- 1. TYPES -----
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  RobustScaler(with_centering=True))
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer(
        [("num", num_pipe, num_cols),
         ("cat", cat_pipe, cat_cols)],
        remainder="drop"
    )

    # --------------------------------------------------------- 2. TIME SPLIT ----
    test_size = int(np.ceil(len(df) * TEST_FRAC))
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # ----------------------------------------------------- 3. FIT‑TRANSFORM -----
    X_train_ready = prep.fit_transform(X_train)
    X_test_ready  = prep.transform(X_test)

    cols_ready = prep.get_feature_names_out()
    X_train_ready = pd.DataFrame(X_train_ready, columns=cols_ready, index=X_train.index)
    X_test_ready  = pd.DataFrame(X_test_ready,  columns=cols_ready, index=X_test.index)

    X_train_ready.to_csv(out_data / "X_train.csv")
    X_test_ready.to_csv(out_data / "X_test.csv")
    y_train.to_csv(out_data / "y_train.csv")
    y_test.to_csv(out_data / "y_test.csv")
    joblib.dump(prep, out_data / "prep.pkl")
    print(f"✓  Dataset saved: {X_train_ready.shape=}  {X_test_ready.shape=}")

    # ------------------------------------------------------- 4. BASE MODELS -----
    models = {
        "RF":  RandomForestRegressor(
                  n_estimators=600, max_depth=None,
                  max_features="sqrt", min_samples_leaf=10,
                  n_jobs=-1, random_state=42),
        "XGB": XGBRegressor(
                  n_estimators=900, learning_rate=0.03,
                  max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=1.5, reg_lambda=1.0,
                  objective='reg:squarederror',
                  n_jobs=-1, random_state=42)
    }

    results = {}
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    for name, est in models.items():
        pipe = Pipeline([("prep", prep), ("model", est)])
        mae_folds = []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            pred = pipe.predict(X_train.iloc[val_idx])
            mae = mean_absolute_error(y_train.iloc[val_idx], pred)
            mae_folds.append(mae)
            print(f"{name}  fold {fold+1}/{N_SPLITS}  MAE={mae: .3f}")
        results[name] = round(float(np.mean(mae_folds)), 4)

        # fit on full train + save
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, out_model / f"{name.lower()}_model.pkl")

        # feature importance
        if name == "RF":
            imp = pipe.named_steps["model"].feature_importances_
        else:
            booster = pipe.named_steps["model"].get_booster()
            fmap    = pipe.named_steps["prep"].get_feature_names_out()
            imp_dict = booster.get_score(importance_type="gain")
            imp = np.array([imp_dict.get(f, 0) for f in fmap])

        pd.Series(imp, index=cols_ready)\
          .sort_values(ascending=False)\
          .to_csv(out_model / f"{name.lower()}_feat_imp.csv")

    # ------------------------------------------------------- 5.  SAVE METRICS ---
    with open(out_model / "cv_mae.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("✓  CV‑MAE:", results)
