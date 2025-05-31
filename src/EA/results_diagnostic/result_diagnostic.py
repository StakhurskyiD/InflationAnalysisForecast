import joblib, pandas as pd, matplotlib
matplotlib.use("Agg")          # або "TkAgg" якщо потрібне живе вікно
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# 1) Pipeline
xgb = joblib.load("models/xgb_model.pkl")

# 2) RAW дані
raw = pd.read_csv("processed_quarter_features.csv",
                  parse_dates=['date_q'], index_col='date_q')
X_raw = raw.drop(columns=["inflation_index_t+1", "date", "quarter_period"])
y_true = raw["inflation_index_t+1"].iloc[-4:]   # hold‑out

# 3) Прогноз
y_hat_full = pd.Series(xgb.predict(X_raw),
                       index=raw.index, name="pred")
# графік останніх 4 кварталів
ax = y_hat_full.iloc[-4:].plot(label="XGB pred", marker="o")
y_true.plot(ax=ax, label="Fact", marker="s")
ax.legend(); ax.set_title("Hold‑out: факт vs прогноз")
plt.tight_layout(); plt.savefig("fig_pred_vs_fact.png", dpi=150)
print("✓  Графік записаний → fig_pred_vs_fact.png")


# import joblib, pandas as pd, matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error
#
# # 1)  Завантажуємо пайплайн
# rf  = joblib.load("models/rf_model.pkl")
# xgb = joblib.load("models/xgb_model.pkl")
#
# # 2)  Графік "факт‑vs‑прогноз" на hold‑out
# raw = pd.read_csv("processed_quarter_features.csv",
#                   parse_dates=['date_q'], index_col='date_q')
# y_true = raw["inflation_index_t+1"].iloc[-4:]                   # наш hold‑out
# y_hat = pd.Series(xgb.predict(raw.drop(columns=["inflation_index_t+1",
#                                                  "date", "quarter_period"])),
#                    index=raw.index, name="pred")
# y_hat.iloc[-4:].plot(label="XGB pred");  y_true.plot(label="Fact")
# plt.legend(); plt.show()
#
# # 3)  SHAP‑важливість для XGB (додатково)
# import shap
# explainer = shap.TreeExplainer(xgb.named_steps["model"])
# shap_vals = explainer.shap_values(
#                xgb.named_steps["prep"].transform(
#                    raw.drop(columns=["inflation_index_t+1","date","quarter_period"])
#                ))
# shap.summary_plot(shap_vals, features=xgb.named_steps["prep"]
#                                   .transform(raw.drop(columns=[...])),
#                   feature_names=xgb.named_steps["prep"].get_feature_names_out())
