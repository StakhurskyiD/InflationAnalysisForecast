###############################################################################
#  SCALE + TIME SPLIT  :  works for trees, Hansen, NN                        #
# -------------------------------------------------------------------------- #
#  INPUT : processed_quarter_features.csv  (ваш файл після FE)              #
#  OUTPUT:                                                                  #
#          •  X_train_scaled.csv, y_train.csv                                #
#          •  X_test_scaled.csv,  y_test.csv                                 #
#          •  scaler_pipeline.pkl  (ColumnTransformer+RobustScaler)          #
###############################################################################
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

RAW_FE = Path("../train/processed_quarter_features.csv")   # <- ваш файл
TARGET = "inflation_index_t+1"                         # <- ваша ціль

# --------------------------------------------------------------------------- #
# 1.   LOAD
# --------------------------------------------------------------------------- #
df = pd.read_csv(RAW_FE, parse_dates=['date_q'], index_col='date_q')

#  беремо всі фічі, крім таргету
X = df.drop(columns=[TARGET])
y = df[TARGET]

# --------------------------------------------------------------------------- #
# 2.   COLUMN GROUPS
# --------------------------------------------------------------------------- #
NUM_COLS = X.select_dtypes('number').columns.tolist()
CAT_COLS = X.select_dtypes('object').columns.tolist()   #  має бути порожній

preprocess = ColumnTransformer(
    [
        ("scale", RobustScaler(with_centering=True), NUM_COLS),
        ("ohe", OneHotEncoder(handle_unknown='ignore'), CAT_COLS)
    ],
    remainder='drop'
)

# огортаємо у Pipeline, щоб .fit/.transform було «в одному місці»
scaler_pipe = Pipeline(steps=[("prep", preprocess)])

# --------------------------------------------------------------------------- #
# 3.   TRAIN / TEST  split  (останній 20 %)
# --------------------------------------------------------------------------- #
#  хронологічний розподіл – без перемішування
test_size = int(np.ceil(len(df) * 0.20))
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

#  fit на train, transform – на обох
X_train_sc = scaler_pipe.fit_transform(X_train)
X_test_sc  = scaler_pipe.transform(X_test)

#  повертаємо у DataFrame, щоб не загубити імена колонок
feat_names = scaler_pipe.get_feature_names_out()
X_train_sc = pd.DataFrame(X_train_sc, columns=feat_names, index=X_train.index)
X_test_sc  = pd.DataFrame(X_test_sc,  columns=feat_names, index=X_test.index)

# --------------------------------------------------------------------------- #
# 4.   SAVE
# --------------------------------------------------------------------------- #
Path("dataset_split").mkdir(exist_ok=True)

X_train_sc.to_csv("dataset_split/X_train_scaled.csv")
X_test_sc.to_csv("dataset_split/X_test_scaled.csv")
y_train.to_csv("dataset_split/y_train.csv")      # 1‑колонний csv
y_test.to_csv("dataset_split/y_test.csv")

joblib.dump(scaler_pipe, "dataset_split/scaler_pipeline.pkl")

print("✓  Масштабування завершено:",
      f"\n   train: {X_train_sc.shape}  test: {X_test_sc.shape}",
      "\n   Файли записані у  dataset_split/")
