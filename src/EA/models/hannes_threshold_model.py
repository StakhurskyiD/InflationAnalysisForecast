# hansen_threshold_plus.py  ─────────────────────────────────────────────
from __future__ import annotations
import io, json, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
import statsmodels.api as sm
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# ---------- core --------------------------------------------------------
def run_hansen(raw_csv: Path, prep_pkl: Path, *,
                    target: str,
                    th_var: str,
                    extra_reg: list[str] = None,
                    out_dir: Path,
                    boot: int = 2000,
                    n_jobs: int = -1) -> None:
    """
    Hansen (2000) one-threshold with
    • optional extra regressors
    • percentile grid 5-95%
    • HAC-robust SE
    • parallel wild-bootstrap
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df   = pd.read_csv(raw_csv, parse_dates=['date_q'], index_col='date_q')
    prep = joblib.load(prep_pkl)

    X_raw = df.drop(columns=[target])
    y     = df[target].values
    X_std = pd.DataFrame(prep.transform(X_raw),
                         columns=prep.get_feature_names_out(),
                         index=X_raw.index)

    if th_var not in X_std.columns:
        raise KeyError(f"{th_var} not found in preprocessed columns")

    # base + extra regressors
    extra_reg = extra_reg or []
    use_cols  = [th_var] + [c for c in extra_reg if c in X_std.columns]
    X_base    = X_std[use_cols].values
    Z         = X_std[th_var].values

    # grid 5-95 percentiles (500 knots)
    grid = np.quantile(Z, np.linspace(.05, .95, 500))

    def rss_gamma(g):
        R = (Z > g).astype(float)
        return sm.OLS(y, sm.add_constant(np.c_[X_base, R])).fit(
                     cov_type="HAC", cov_kwds={"maxlags":4}).ssr

    rss        = np.array([rss_gamma(g) for g in grid])
    gamma_hat  = float(grid[np.argmin(rss)])

    # ----- parallel wild-bootstrap --------------------------------------
    from joblib import Parallel, delayed
    rng = np.random.default_rng(42)

    def one_rep(seed):
        ν  = rng.integers(0, 2, size=len(y))*2 - 1         # ±1
        y_star = y + ν * (y - y.mean())                    # wild
        rss_star = [rss_gamma(g) for g in grid]
        return min(rss_star) - rss_star[np.argmin(rss)]

    supW_obs = min(rss) - rss[np.argmin(rss)]
    stats = Parallel(n_jobs=n_jobs)(
        delayed(one_rep)(i) for i in range(boot))
    p_val = np.mean(np.array(stats) < supW_obs)

    # ----- plots ---------------------------------------------------------
    _save_png(out_dir/"rss_curve.png",
              _plot_rss(grid, rss, gamma_hat))
    _save_png(out_dir/"scatter_regimes.png",
              _plot_scatter(Z, y, gamma_hat, th_var, target))

    json.dump({"gamma_hat": gamma_hat,
               "p_value":   float(p_val),
               "th_var":    th_var,
               "boot_rep":  boot,
               "extra_reg": use_cols[1:]},
              open(out_dir/"hansen_summary.json","w"), indent=2, ensure_ascii=False)

    print(f"γ̂={gamma_hat:.3f};  p={p_val:.4f}  |  artefacts → {out_dir.resolve()}")

# ---------- tiny helpers -----------------------------------------------
def _plot_rss(grid, rss, γ):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(grid, rss); ax.axvline(γ, c='r', ls='--')
    ax.set(title="RSS(γ) – threshold search",
           xlabel="γ (standardised)", ylabel="Residual SS")
    fig.tight_layout(); return fig

def _plot_scatter(Z,y,γ,xlab,ylab):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.scatter(Z, y, c=(Z>γ), cmap="coolwarm", s=50, alpha=.8)
    ax.axvline(γ, color='k', ls='--')
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title("Regimes by threshold"); fig.tight_layout(); return fig

def _save_png(path: Path, fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160)
    plt.close(fig); path.write_bytes(buf.getvalue())

# ---------- CLI ---------------------------------------------------------
if __name__ == "__main__":
    import argparse, warnings
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--prep",required=True, type=Path)
    p.add_argument("--target", default="inflation_index_t+1")
    p.add_argument("--th-var", required=True)
    p.add_argument("--extra-reg", nargs="*", default=[],
                   help="додаткові стандартизовані регресори")
    p.add_argument("--out", type=Path, default="hansen_out_plus")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--jobs", type=int, default=-1)
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    run_hansen(args.raw, args.prep,
                    target   = args.target,
                    th_var   = args.th_var,
                    extra_reg= args.extra_reg,
                    out_dir  = args.out,
                    boot     = args.boot,
                    n_jobs   = args.jobs)

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# hansen_threshold.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Оцінка однопорогової моделі Hansen (2000) для квартальних даних.
# – підтримує будь-яку стандартизовану змінну-поріг;
# – wild-bootstrap для малих вибірок;
# – автоматично тягне prep.pkl, щоб не плодити дубльовані трансформації.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Usage
# ------
# python hansen_threshold.py \
#        --raw  src/research_data/processed_data/quarterly_data_with_features.csv \
#        --prep src/research_data/processed_data/prep.pkl \
#        --target inflation_index_t+1 \
#        --th-var inflation_gap \
#        --out  hansen_out \
#        --boot 1000
# """
# from __future__ import annotations
# import argparse, json, warnings
# import io
# from pathlib import Path
#
# import joblib, numpy as np, pandas as pd
# from statsmodels.regression.linear_model import OLS
# from statsmodels.tools.tools import add_constant
# from tqdm.auto import tqdm
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
#
# # ────────────────────────────── utils ──────────────────────────────────── #
# def wild_bootstrap_rss(y, X_base, Z, grid, gamma_hat, n_rep=999,
#                        random_state: int | None = None) -> float:
#     """Повертає p-value для sup-Wald статистики через wild bootstrap."""
#     rng = np.random.default_rng(random_state)
#
#     def rss(y_, R_):
#         return OLS(y_, add_constant(np.c_[X_base, R_])).fit().ssr
#
#     rss0 = rss(y, (Z > gamma_hat).astype(float))
#     supW = np.min([rss(y, (Z > g).astype(float)) for g in grid]) - rss0
#
#     greater = 0
#     for _ in tqdm(range(n_rep), desc="wild-bootstrap", leave=False):
#         nu = rng.choice([-1, 1], size=len(y))
#         y_star = y + (y - X_base @ np.linalg.lstsq(X_base, y, rcond=None)[0]) * nu
#         rss_star = [rss(y_star, (Z > g).astype(float)) for g in grid]
#         supW_star = min(rss_star) - rss_star[grid.tolist().index(gamma_hat)]
#         greater += supW_star < supW
#     return greater / n_rep
#
#
# # ────────────────────────────── main ------------------------------------ #
# def run_hansen(raw_csv: Path, prep_pkl: Path,
#                target: str, th_var: str,
#                out_dir: Path, boot: int = 999) -> None:
#
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     # 1. LOAD ----------------------------------------------------------------
#     df = pd.read_csv(raw_csv, parse_dates=["date_q"], index_col="date_q")
#     prep = joblib.load(prep_pkl)
#
#     X_raw = df.drop(columns=[target])
#     y = df[target].values
#
#     X_std = pd.DataFrame(
#         prep.transform(X_raw),
#         columns=prep.get_feature_names_out(),
#         index=X_raw.index
#     )
#
#     if th_var not in X_std.columns:
#         raise KeyError(f"Порогова змінна '{th_var}' відсутня серед prep-фіч!")
#
#     Z = X_std[th_var].values
#     X_base = X_std[[th_var]].values          # модель: y = β0 + β1·Z + δ·R + ε
#
#     # 2. Grid search гав-гамма ----------------------------------------------
#     trim = 0.15                              # Hansenʼs рекомендація
#     lower, upper = np.quantile(Z, [trim, 1-trim])
#     grid = np.linspace(lower, upper, 250)
#
#     def rss_gamma(g):
#         R = (Z > g).astype(float)
#         return OLS(y, add_constant(np.c_[X_base, R])).fit().ssr
#
#     rss = np.array([rss_gamma(g) for g in grid])
#     gamma_hat = float(grid[np.argmin(rss)])
#
#     # 3. Wild bootstrap p-value ---------------------------------------------
#     p_val = wild_bootstrap_rss(y, X_base, Z, grid, gamma_hat,
#                                n_rep=boot, random_state=42)
#
#     # 4. Save results --------------------------------------------------------
#     (out_dir / "rss_curve.png").write_bytes(
#         _plot_rss(grid, rss, gamma_hat).getvalue()
#     )
#     (out_dir / "scatter_regimes.png").write_bytes(
#         _plot_scatter(Z, y, gamma_hat, th_var, target).getvalue()
#     )
#
#     res = {"gamma_hat": gamma_hat, "p_value": p_val,
#            "th_var": th_var, "boot_rep": boot}
#     with open(out_dir / "hansen_summary.json", "w") as fh:
#         json.dump(res, fh, indent=2, ensure_ascii=False)
#
#     print("————————————————————————————")
#     print(f"γ̂ = {gamma_hat:.3f}  |  p-value = {p_val:.4f}  (wild bootstrap, {boot})")
#     print("Artefacts saved to →", out_dir.resolve())
#
#
# # ─────────────────────────── plotting helpers ──────────────────────────── #
# def _plot_rss(grid, rss, gamma_hat):
#     plt.figure(figsize=(7, 4))
#     plt.plot(grid, rss, lw=1.4)
#     plt.axvline(gamma_hat, color="red", ls="--")
#     plt.title("RSS(γ) – Hansen threshold search")
#     plt.xlabel("γ (standardised)")
#     plt.ylabel("Residual sum-of-squares")
#     plt.tight_layout()
#
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", dpi=160)
#     plt.close()
#     buf.seek(0)                # → на початок, щоб можна було читати
#     return buf                 # готові PNG-байти
#
#
# def _plot_scatter(Z, y, gamma_hat, xlab, ylab):
#     plt.figure(figsize=(7, 4))
#     plt.scatter(Z, y, c=(Z > gamma_hat), cmap="coolwarm",
#                 s=45, alpha=0.85)
#     plt.axvline(gamma_hat, color="black", ls="--")
#     plt.xlabel(xlab)
#     plt.ylabel(ylab)
#     plt.title("Two-regime scatter")
#     plt.tight_layout()
#
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", dpi=160)
#     plt.close()
#     buf.seek(0)
#     return buf
#
#
# # ───────────────────────────── CLI entry ───────────────────────────────── #
# if __name__ == "__main__":
#     p = argparse.ArgumentParser("Hansen one-threshold estimation")
#     p.add_argument("--raw", required=True, type=Path,
#                    help="quarterly_data_with_features.csv")
#     p.add_argument("--prep", required=True, type=Path,
#                    help="prep.pkl (ColumnTransformer)")
#     p.add_argument("--target", default="inflation_index_t+1")
#     p.add_argument("--th-var", required=True,
#                    help="Назва стандартизованої змінної-порога (після prep)")
#     p.add_argument("--out", type=Path, default="hansen_out")
#     p.add_argument("--boot", type=int, default=999,
#                    help="bootstrap repetitions")
#     args = p.parse_args()
#
#     warnings.filterwarnings("ignore")
#     run_hansen(args.raw, args.prep, args.target,
#                args.th_var, args.out, boot=args.boot)
#
#
