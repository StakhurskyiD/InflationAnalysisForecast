"""
RAW  →  prep.pkl  →  RF + XGB
-----------------------------------------
• читає макро‑фічі без попередньої обробки
• підключає ColumnTransformer (імпутер+OHE+скейл)
• проводить 5‑фолд TimeSeries CV
• фітить RandomForest та XGBoost на full‑train
• зберігає *.pkl  і  feature‑importance
"""
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ------------------------ CONFIG ------------------------------------------
RAW_FILE   = Path('processed_quarter_features.csv')
PREP_FILE  = Path('prep.pkl')        # уже навчений ColumnTransformer
TARGET     = "inflation_index_t+1"
DROP       = ["date", "quarter_period"]            # raw‑колонки, що не потрібні
TEST_FRAC  = 0.20                                  # останні 20 % → hold‑out
N_SPLITS   = 5                                     # CV усередині train
MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ------------------------ 0. LOAD RAW -------------------------------------
df = pd.read_csv(RAW_FILE, parse_dates=["date_q"], index_col="date_q")
X = df.drop(columns=[TARGET] + DROP)
y = df[TARGET]

# ------------------------ 1. LOAD PREPROCESSOR ----------------------------
prep: ColumnTransformer = joblib.load(PREP_FILE)
print("✓ prep.pkl завантажено — типи трансформерів:",
      [name for name, _, _ in prep.transformers])

# ------------------------ 2. TIME SPLIT -----------------------------------
test_size = int(np.ceil(len(df) * TEST_FRAC))
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

# ------------------------ 3. МОДЕЛІ ---------------------------------------
models = {
    "RF": RandomForestRegressor(
            n_estimators=600, max_depth=None,
            max_features="sqrt", min_samples_leaf=10,
            n_jobs=-1, random_state=42),
    "XGB": XGBRegressor(
            n_estimators=900, learning_rate=0.03,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.5, reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=-1, random_state=42)
}

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
cv_mae = {}

for name, est in models.items():
    pipe = Pipeline([("prep", prep), ("model", est)])

    # ---- CV inside train --------------------------------------------------
    fold_mae = []
    for k, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = pipe.predict(X_train.iloc[val_idx])
        mae  = mean_absolute_error(y_train.iloc[val_idx], pred)
        fold_mae.append(mae)
        print(f"{name} fold {k+1}/{N_SPLITS}   MAE = {mae: .3f}")
    cv_mae[name] = float(np.mean(fold_mae))
    print(f"{name} mean CV‑MAE: {cv_mae[name]: .3f}")

    # ---- fit on full train + save ----------------------------------------
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_DIR / f"{name.lower()}_model.pkl")

    # ---- feature importance ---------------------------------------------
    if name == "RF":
        imp = pipe.named_steps["model"].feature_importances_
    else:
        booster  = pipe.named_steps["model"].get_booster()
        fmap     = pipe.named_steps["prep"].get_feature_names_out()
        imp_dict = booster.get_score(importance_type="gain")
        imp      = np.array([imp_dict.get(f, 0.0) for f in fmap])

    pd.Series(imp,
              index=pipe.named_steps["prep"].get_feature_names_out())\
      .sort_values(ascending=False)\
      .to_csv(MODEL_DIR / f"{name.lower()}_feat_imp.csv")

# ------------------------ 4. HOLD‑OUT METRIC ------------------------------
pipe_rf  = joblib.load(MODEL_DIR / "rf_model.pkl")
pipe_xgb = joblib.load(MODEL_DIR / "xgb_model.pkl")
mae_test = {
    "RF":  mean_absolute_error(y_test,  pipe_rf.predict(X_test)),
    "XGB": mean_absolute_error(y_test, pipe_xgb.predict(X_test))
}

# ------------------------ 5. SAVE METRICS ---------------------------------
with open(MODEL_DIR / "cv_mae.json", "w") as fp:
    json.dump({"cv_mae": cv_mae, "test_mae": mae_test}, fp, indent=2)

print("\n✓ Training complete.  Models & metrics збережено:")
print("  CV MAE :", cv_mae)
print("  Test MAE:", mae_test)
