import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chaospy as cp

folder  = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(folder, "outputs")
np.random.seed(42)

"""Reservoir and PCE configuration"""

params = pd.read_csv(os.path.join(out_dir, "F14_parameters.csv"), index_col="Parameter")["Value"]
Pi = float(params["Pi"])

ct0 = 15.0e-6
PCE_ORDER, SPARSE_LEVEL, N_MC = 3, 3, 10_000
ct_lo, ct_hi     = ct0 * 0.80, ct0 * 1.20
zeta_lo, zeta_hi = 1.0 / ct_hi, 1.0 / ct_lo

DRAWDOWN_MIN = 100.0   # psi -- same threshold 07 uses; reused as the BMA sigmoid's center/scale

"""Load real F-14 record and the PI-branch PCE results already computed by 07"""

df  = pd.read_csv(os.path.join(out_dir, "Volve_F14_Pavg_daily.csv"), parse_dates=["date"])
uq  = pd.read_csv(os.path.join(out_dir, "UQ_results_real.csv"), parse_dates=["date"])
n_days = len(df)

q_arr    = df["q_oil_STB_day"].values.astype(float)
P_arr    = df["BHP_psi"].values.astype(float)
Pavg_ref = df["Pavg_MBE_psi"].values.astype(float)
st_arr   = np.where(df["producing_day"].values == 1, 0, np.where(df["shutin_day"].values == 1, 2, -1))
Np_arr   = np.cumsum(q_arr)
N_ooip   = float(Np_arr[-1]) / (ct0 * (Pi - float(np.nanmin(Pavg_ref))))
dd_arr   = Pavg_ref - P_arr   # drawdown, psi

mu_PI  = uq["Pavg_mean_pce"].values.astype(float)
sig_PI = uq["Pavg_std_pce"].values.astype(float)

print(f"Loaded {n_days} days. PI-branch PCE stats taken from UQ_results_real.csv (producing days).")

"""PCE infrastructure -- same standardized d=4 joint as 07"""

joint = cp.J(cp.Normal(0, 2.0), cp.Normal(0, 1.0), cp.Uniform(-1, 1), cp.Uniform(-1, 1))
poly           = cp.generate_expansion(PCE_ORDER, joint, normed=True)
nodes, weights = cp.generate_quadrature(SPARSE_LEVEL, joint, rule="gaussian", sparse=True)
poly_at_nodes_w = np.array(poly(*nodes), dtype=float) * weights[np.newaxis, :]
mc              = joint.sample(N_MC, rule="random", seed=42)
poly_at_mc      = np.array(poly(*mc), dtype=float)
zeta_phys_nodes = zeta_lo + (nodes[3] + 1.0) / 2.0 * (zeta_hi - zeta_lo)

"""MBE branch evaluated on every day with valid cumulative production"""

mu_MBE  = np.full(n_days, np.nan)
sig_MBE = np.full(n_days, np.nan)
for i in range(n_days):
    Np_t = float(Np_arr[i])
    if not np.isfinite(Np_t):
        continue
    model_evals = Pi - Np_t * zeta_phys_nodes / N_ooip
    coeffs = poly_at_nodes_w @ model_evals
    pce_mc = np.clip(coeffs @ poly_at_mc, 0.0, Pi)
    mu_MBE[i]  = np.mean(pce_mc)
    sig_MBE[i] = np.std(pce_mc)

print(f"MBE branch computed on {np.isfinite(mu_MBE).sum()} days (all days with cumulative Np).")

"""Drawdown-sigmoid BMA weights"""

w_PI  = 1.0 / (1.0 + np.exp(-(dd_arr - DRAWDOWN_MIN) / DRAWDOWN_MIN))
w_MBE = 1.0 - w_PI

pi_mask = (st_arr == 0) & np.isfinite(mu_PI) & np.isfinite(mu_MBE)

mu_BMA  = np.full(n_days, np.nan)
sig_BMA = np.full(n_days, np.nan)
mu_BMA[pi_mask] = w_PI[pi_mask] * mu_PI[pi_mask] + w_MBE[pi_mask] * mu_MBE[pi_mask]
# two-component Gaussian-mixture variance: E[X^2] - E[X]^2, per component E[X^2] = sigma^2 + mu^2
ex2 = w_PI[pi_mask] * (sig_PI[pi_mask]**2 + mu_PI[pi_mask]**2) + w_MBE[pi_mask] * (sig_MBE[pi_mask]**2 + mu_MBE[pi_mask]**2)
sig_BMA[pi_mask] = np.sqrt(np.maximum(ex2 - mu_BMA[pi_mask]**2, 0.0))

"""Verify the mixture behaves as intended and quantify what it changes"""

disagree = np.abs(mu_PI[pi_mask] - Pavg_ref[pi_mask])
print(f"\nProducing days in BMA blend: {pi_mask.sum()}")
print(f"w_PI  range: [{w_PI[pi_mask].min():.3f}, {w_PI[pi_mask].max():.3f}]  mean={w_PI[pi_mask].mean():.3f}")
print(f"Median |mu_PI - Pavg_ref_MBE| (inter-model disagreement): {np.median(disagree):.1f} psi")
print(f"Median sigma_PI (reported PI-only uncertainty):      {np.median(sig_PI[pi_mask]):.2f} psi")
print(f"Median sigma_BMA (blended uncertainty):               {np.median(sig_BMA[pi_mask]):.1f} psi")
print(f"Ratio disagreement / sigma_PI  (current framework):   {np.median(disagree)/np.median(sig_PI[pi_mask]):.1f}x")
print(f"Ratio disagreement / sigma_BMA (with BMA layer):      {np.median(disagree)/np.median(sig_BMA[pi_mask]):.1f}x")

early = dd_arr[pi_mask] > 500
late  = dd_arr[pi_mask] < -100
print(f"\nHigh-drawdown days (dd>500psi, n={early.sum()}): mean w_PI={w_PI[pi_mask][early].mean():.3f}, "
      f"mean |mu_BMA-mu_PI|={np.abs(mu_BMA[pi_mask][early]-mu_PI[pi_mask][early]).mean():.1f} psi")
print(f"Low/negative-drawdown days (dd<-100psi, n={late.sum()}): mean w_PI={w_PI[pi_mask][late].mean():.3f}, "
      f"mean |mu_BMA-mu_MBE|={np.abs(mu_BMA[pi_mask][late]-mu_MBE[pi_mask][late]).mean():.1f} psi")

"""Save outputs"""

pd.DataFrame({
    "day": np.arange(n_days), "date": df["date"].values, "drawdown_psi": dd_arr,
    "mu_PI": mu_PI, "sigma_PI": sig_PI, "mu_MBE": mu_MBE, "sigma_MBE": sig_MBE,
    "w_PI": w_PI, "w_MBE": w_MBE, "mu_BMA": mu_BMA, "sigma_BMA": sig_BMA,
}).to_csv(os.path.join(out_dir, "BMA_results.csv"), index=False)
print("\nSaved: outputs/BMA_results.csv")

"""Figure 15: BMA weight vs drawdown and over field life.
Figure 16: blended mean vs component models and reported uncertainty."""

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 300})

dates_pd = pd.to_datetime(df["date"].values)

# Figure 15: BMA weight vs drawdown (A) and over field life (B)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Drawdown-Sigmoid BMA Weight Function", fontsize=11, fontweight="bold")

ax = axes[0]
dd_x = np.linspace(-800, 800, 400)
w_x  = 1.0 / (1.0 + np.exp(-(dd_x - DRAWDOWN_MIN) / DRAWDOWN_MIN))
ax.plot(dd_x, w_x, color="#08306b", lw=2)
ax.axvline(DRAWDOWN_MIN, ls="--", color="gray", lw=0.8, label=f"center = {DRAWDOWN_MIN:.0f} psi")
ax.axhline(0.5, ls=":", color="gray", lw=0.8)
ax.set(xlabel="Drawdown (psi)", ylabel=r"$w_{PI}$", title="(A) BMA Weight vs Drawdown")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(dates_pd[pi_mask], w_PI[pi_mask], s=3, alpha=0.4, color="#08306b")
ax.set(xlabel="Date", ylabel=r"$w_{PI}$", title="(B) PI Weight Over Field Life", ylim=(-0.05, 1.05))
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig15_bma_weight.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig15_bma_weight.png")

# Figure 16: blended mean vs component models (A) and reported uncertainty (B)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Bayesian Model Averaging of PI and MBE Estimators: Mean and Uncertainty", fontsize=11, fontweight="bold")

ax = axes[0]
ax.plot(dates_pd[pi_mask], mu_PI[pi_mask], color="#d62728", lw=0.7, alpha=0.6, label="PI only")
ax.plot(dates_pd[pi_mask], mu_MBE[pi_mask], color="#1f77b4", lw=0.7, alpha=0.6, label="MBE reference")
ax.plot(dates_pd[pi_mask], mu_BMA[pi_mask], color="black", lw=1.0, label="BMA blend")
ax.set(xlabel="Date", ylabel="Pavg (psi)", title="(A) Blended Mean vs Component Models")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
PLOT_FLOOR = 1e-2
sig_PI_plot = np.maximum(sig_PI[pi_mask], PLOT_FLOOR)
ax.plot(dates_pd[pi_mask], sig_PI_plot, color="#d62728", lw=0.8, label=r"$\sigma_{PCE}$ (without BMA)")
ax.plot(dates_pd[pi_mask], sig_BMA[pi_mask], color="black", lw=1.0, label=r"$\sigma_{BMA}$")
ax.axhline(PLOT_FLOOR, color="#d62728", lw=0.5, ls=":", alpha=0.5)
ax.set(xlabel="Date", ylabel="Reported uncertainty (psi)", title="(B) Reported Uncertainty: PI-Only vs BMA", yscale="log")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig16_bma_mean_uncertainty.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig16_bma_mean_uncertainty.png")
