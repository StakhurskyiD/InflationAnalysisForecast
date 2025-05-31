# hansen_threshold.py
# ---------------------------------------------------------------
# Estimate Hansen (2000) one‑threshold model + diagnostics plots
# ---------------------------------------------------------------
import warnings, joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")                     # без GUI (максимальна сумісність)
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from tqdm import tqdm
from scipy.stats import norm

RAW      = Path("processed_quarter_features.csv")
PREP     = Path("prep.pkl")               # ColumnTransformer
TARGET   = "inflation_index_t+1"
DROP     = ["date", "quarter_period"]
TH_VAR   = "num__inflation_gap"                         # назва після prep
BOOT_R   = 300                                          # *↑* для дипломної
OUT_DIR  = Path("hansen_out");  OUT_DIR.mkdir(exist_ok=True)

# 1. LOAD & PREP -------------------------------------------------
df_raw = pd.read_csv(RAW, parse_dates=['date_q'], index_col='date_q')
X_raw  = df_raw.drop(columns=[TARGET] + DROP)
y      = df_raw[TARGET]

prep = joblib.load(PREP)

X_std = pd.DataFrame(
        prep.transform(X_raw),
        columns=prep.get_feature_names_out(),
        index=X_raw.index)

Z = X_std[TH_VAR].values              # порогова змінна
grid = np.quantile(Z, np.linspace(0.15, 0.85, 250))   # Hansenʼs «trim»

def rss_given_gamma(gamma):
    R = (Z > gamma).astype(float)
    X = add_constant( np.c_[X_std[TH_VAR], R] )        # простий приклад: 1 регресор + dummy
    return OLS(y, X).fit().ssr

rss = np.array([rss_given_gamma(g) for g in grid])
gamma_hat = float(grid[np.argmin(rss)])
print(f"✓  Оціночний поріг γ̂  =  {gamma_hat: .3f}  (у σ‑одиницях стандартизації)")

# 2. Wild‑bootstrap p‑value -------------------------------------
def wild_boot(rep):
    # ν_i  ~ {‑1, +1} з імовірністю 0.5
    rng = np.random.default_rng(rep)
    Xi  = add_constant( np.c_[X_std[TH_VAR], (Z > gamma_hat).astype(float)] )
    fit = OLS(y, Xi).fit()
    resid = fit.resid
    y_star = fit.fittedvalues + resid * rng.choice([-1, 1], size=len(resid))
    rss_star = [OLS(y_star, add_constant(np.c_[X_std[TH_VAR], (Z > g)] )).fit().ssr
                for g in grid]
    return min(rss_star) - rss_star[grid.tolist().index(gamma_hat)]

stat_orig = min(rss) - rss[grid.tolist().index(gamma_hat)]
boot_stats = np.array([wild_boot(r) for r in tqdm(range(BOOT_R), desc="wild‑bootstrap")])
p_value = np.mean(boot_stats < stat_orig)
print(f"p‑value (wild bootstrap, {BOOT_R} rep): {p_value: .4f}")

# 3. PLOTS -------------------------------------------------------
# 3.1 RSS curve
plt.figure(figsize=(7,4))
plt.plot(grid, rss, lw=1.3)
plt.axvline(gamma_hat, color="red", ls="--", label=f"γ̂ = {gamma_hat: .3f}")
plt.title("RSS(γ) – Hansen threshold search")
plt.xlabel("γ  (standardised inflation_gap)"); plt.ylabel("Residual SS")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "rss_curve.png", dpi=160)

# 3.2 Scatter with regimes
plt.figure(figsize=(7,4))
plt.scatter(Z, y, c=(Z > gamma_hat), cmap="coolwarm", s=45, alpha=0.8)
plt.axvline(gamma_hat, color="black", ls="--")
plt.title("Inflation target vs inflation_gap  (two regimes)")
plt.xlabel("inflation_gap  (σ)"); plt.ylabel("inflation_index t+1")
plt.tight_layout(); plt.savefig(OUT_DIR / "scatter_regimes.png", dpi=160)

print("\n✓  Графіки збережено у :", OUT_DIR.resolve())
