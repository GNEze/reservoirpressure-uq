import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chaospy as cp
from scipy import stats
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import legval

folder  = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(folder, "outputs")
np.random.seed(42)

BAR2PSI = 14.5038
SM3_STB = 6.2898

"""Reservoir and PCE configuration"""

params = pd.read_csv(os.path.join(out_dir, "F14_parameters.csv"), index_col="Parameter")["Value"]
Pi       = float(params["Pi"])
J0       = float(params["J0"])
Np_total = float(params["Np_total"])

ct0        = 15.0e-6
sig_P      = 2.0
sig_q_frac = 0.02
Pavg_final = 200.0 * BAR2PSI   # Volve 8.5-year depletion anchor

J_lo, J_hi   = J0 * 0.85, J0 * 1.15      # +-15%
ct_lo, ct_hi = ct0 * 0.80, ct0 * 1.20    # +-20%
eta_lo, eta_hi   = 1.0 / J_hi, 1.0 / J_lo     # eta = 1/J
zeta_lo, zeta_hi = 1.0 / ct_hi, 1.0 / ct_lo   # zeta = 1/ct

PCE_ORDER    = 3
SPARSE_LEVEL = 3
N_MC         = 10_000
labels = ["ep_P (BHP noise)", "ep_q (flow noise)", "J (PI uncert.)", "ct (compress.)"]

N_ooip = Np_total / (ct0 * (Pi - Pavg_final))

"""Load digital twin data, aggregate to daily"""

df = pd.read_csv(os.path.join(out_dir, "Pavg_estimates.csv"))
df["day"] = (df["time_min"] // 1440).astype(int)
daily = df.groupby("day").agg(
    P_mean=("P_1min_psi", "mean"),
    q_mean=("q_1min_STBm", "mean"),
    Np_cum=("Np_cum_STB", "last"),
    state_mode=("state", lambda x: int(x.mode().iloc[0])),
).reset_index()
daily["q_daily"] = daily["q_mean"] * 1440.0

df_true   = pd.read_csv(os.path.join(out_dir, "digital_twin_true_Pavg.csv"))
Pavg_true = df_true["Pavg_true"].values

n_days = len(daily)
P_arr  = daily["P_mean"].values
q_arr  = daily["q_daily"].values
Np_arr = daily["Np_cum"].values
st_arr = daily["state_mode"].values

print(f"Days: {n_days}  N_ooip: {N_ooip:,.0f} STB")

"""PCE infrastructure: single d=4 joint shared by both forward models"""

def sobol_masks(poly_arr, n_inputs):
    # per-polynomial exponents, not chaospy's shared array-level table
    n_terms = len(poly_arr)
    depends = np.zeros((n_inputs, n_terms), dtype=bool)
    for i in range(n_terms):
        exps = np.array(poly_arr[i].exponents, dtype=int)
        for k in range(n_inputs):
            depends[k, i] = np.any(exps[:, k] > 0)
    n_dep    = depends.sum(axis=0)
    is_const = n_dep == 0
    S1_mask = np.zeros((n_inputs, n_terms), dtype=bool)
    ST_mask = np.zeros((n_inputs, n_terms), dtype=bool)
    for k in range(n_inputs):
        S1_mask[k] = depends[k] & (n_dep == 1)
        ST_mask[k] = depends[k]
    return S1_mask, ST_mask, is_const

joint = cp.J(
    cp.Normal(0, sig_P),        # eps_P  [0]
    cp.Normal(0, 1.0),          # eps_q  [1] -- scaled by sig_q/q(t) per day
    cp.Uniform(eta_lo, eta_hi), # eta = 1/J  [2]
    cp.Uniform(zeta_lo, zeta_hi), # zeta = 1/ct  [3]
)
poly              = cp.generate_expansion(PCE_ORDER, joint, normed=True)
nodes, weights    = cp.generate_quadrature(SPARSE_LEVEL, joint, rule="gaussian", sparse=True)
n_nodes, n_terms  = nodes.shape[1], len(poly)
assert abs(float(np.sum(weights)) - 1.0) < 1e-10

mc              = joint.sample(N_MC, rule="random", seed=42)
poly_at_nodes   = np.array(poly(*nodes), dtype=float)
poly_at_nodes_w = poly_at_nodes * weights[np.newaxis, :]
poly_at_mc      = np.array(poly(*mc), dtype=float)

mc_epi          = np.vstack([np.zeros((2, N_MC)), mc[2:4, :]])   # eps_P=eps_q=0
poly_at_mc_epi  = np.array(poly(*mc_epi), dtype=float)

S1_mask, ST_mask, is_const = sobol_masks(poly, 4)

print(f"PCE: d=4, p={PCE_ORDER}, Smolyak level={SPARSE_LEVEL}, nodes={n_nodes}, terms={n_terms}")

"""Main PCE loop: per-day fit, statistics, Sobol' indices"""

Pavg_mean, Pavg_var, Pavg_std = (np.full(n_days, np.nan) for _ in range(3))
Pavg_p025, Pavg_p975         = np.full(n_days, np.nan), np.full(n_days, np.nan)
Pavg_p025_epi, Pavg_p975_epi = np.full(n_days, np.nan), np.full(n_days, np.nan)
Pavg_p050, Pavg_p250, Pavg_p500, Pavg_p750, Pavg_p950 = (np.full(n_days, np.nan) for _ in range(5))
Pavg_skew, Pavg_kurt = np.full(n_days, np.nan), np.full(n_days, np.nan)
Sobol_S1 = np.full((n_days, 4), np.nan)
Sobol_ST = np.full((n_days, 4), np.nan)
pce_method = np.full(n_days, "none", dtype=object)

for i in range(n_days):
    s = st_arr[i]
    try:
        if s == 0:
            # Stable production: PI model, unclipped (no Gibbs risk) -- Eq. 1
            Pwf_t, q_t = float(P_arr[i]), float(q_arr[i])
            if not (np.isfinite(Pwf_t) and np.isfinite(q_t) and q_t > 0):
                continue
            sig_q_t = sig_q_frac * q_t
            eps_q_n = nodes[1] * sig_q_t
            model_evals = Pwf_t + nodes[0] + (q_t + eps_q_n) * nodes[2]
            coeffs = poly_at_nodes_w @ model_evals

            # Physical bound Pavg<=Pi applied post hoc to MC draws, not inside
            # the projection to avoid the Gibbs phenomenon (Sec. 3.4).
            pce_mc     = np.clip(coeffs @ poly_at_mc,     0.0, Pi)
            pce_mc_epi = np.clip(coeffs @ poly_at_mc_epi, 0.0, Pi)
            pce_method[i] = "PI"

        elif s == 2:
            # Pressure buildup: MBE model, linear in zeta -- Eq. 2
            Np_t = float(Np_arr[i])
            if not np.isfinite(Np_t):
                continue
            model_evals = Pi - Np_t * nodes[3] / N_ooip
            coeffs = poly_at_nodes_w @ model_evals
            pce_mc     = np.clip(coeffs @ poly_at_mc, 0.0, Pi)
            pce_mc_epi = pce_mc   # no sensor noise in MBE -> total CI = epistemic CI
            pce_method[i] = "MBE"
        else:
            continue   # transient / injection: interpolated below

        Pavg_mean[i] = np.mean(pce_mc)
        Pavg_var[i]  = np.var(pce_mc)
        Pavg_std[i]  = np.sqrt(Pavg_var[i])
        Pavg_p025[i], Pavg_p975[i]         = np.percentile(pce_mc,     [2.5, 97.5])
        Pavg_p025_epi[i], Pavg_p975_epi[i] = np.percentile(pce_mc_epi, [2.5, 97.5])
        Pavg_p050[i], Pavg_p250[i], Pavg_p500[i], Pavg_p750[i], Pavg_p950[i] = \
            np.percentile(pce_mc, [5, 25, 50, 75, 95])
        Pavg_skew[i] = stats.skew(pce_mc, bias=False)
        Pavg_kurt[i] = stats.kurtosis(pce_mc, bias=False)

        # Sobol' indices from the smooth (unclipped) PCE coefficients 
        var_pce = float(np.dot(coeffs[~is_const], coeffs[~is_const]))
        if var_pce > 1e-12:
            c2 = coeffs ** 2
            Sobol_S1[i] = np.clip(S1_mask @ c2 / var_pce, 0.0, 1.0)
            Sobol_ST[i] = np.clip(ST_mask @ c2 / var_pce, 0.0, 1.0)
    except Exception:
        pass

def interp_nan(arr):
    nans = np.isnan(arr)
    if 0 < nans.sum() < len(arr):
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr

for col in (Pavg_mean, Pavg_var, Pavg_std, Pavg_p025, Pavg_p975, Pavg_p025_epi, Pavg_p975_epi):
    interp_nan(col)

n_pi, n_mbe, n_none = (pce_method == "PI").sum(), (pce_method == "MBE").sum(), (pce_method == "none").sum()
print(f"PI days={n_pi}  MBE days={n_mbe}  interpolated={n_none}")

"""PCE convergence study: p=1..4, clipped vs unclipped, vs MC reference (day 192)"""

# Reference day: highest-flow stable day where the unclipped model exceeds Pi
# at eta_hi, i.e. where the clip is genuinely active.
stable_idx = np.where(st_arr == 0)[0]
ref_day_conv = None
for d in stable_idx[np.argsort(q_arr[stable_idx])[::-1]]:
    q_d, Pwf_d = float(q_arr[d]), float(P_arr[d])
    if np.isfinite(Pwf_d) and np.isfinite(q_d) and q_d > 0 and Pwf_d + q_d * eta_hi > Pi:
        ref_day_conv = int(d)
        break
if ref_day_conv is None:
    ref_day_conv = int(stable_idx[len(stable_idx) // 2])

Pwf_cv   = float(P_arr[ref_day_conv])
q_cv     = float(q_arr[ref_day_conv])
sig_q_cv = sig_q_frac * max(q_cv, 1.0)
print(f"Convergence reference day: {ref_day_conv}  (Pwf={Pwf_cv:.2f} psi, q={q_cv:.1f} STB/day)")

conv_rows = []
for order in (1, 2, 3, 4):
    poly_o    = cp.generate_expansion(order, joint, normed=True)
    model_raw = Pwf_cv + nodes[0] + (q_cv + nodes[1] * sig_q_cv) * nodes[2]

    poly_at_n_o  = np.array(poly_o(*nodes), dtype=float)
    poly_at_nw_o = poly_at_n_o * weights[np.newaxis, :]
    coeffs_noclip = poly_at_nw_o @ model_raw
    coeffs_clip   = poly_at_nw_o @ np.clip(model_raw, 0.0, Pi)

    is_c_o  = np.all(np.array(poly_o.exponents, dtype=int) == 0, axis=1)
    c_idx   = int(np.where(is_c_o)[0][0])
    nc_mask = ~is_c_o

    conv_rows.append({
        "p": order, "terms": len(poly_o),
        "mean_noclip": float(coeffs_noclip[c_idx]),
        "var_noclip": float(max(np.dot(coeffs_noclip[nc_mask], coeffs_noclip[nc_mask]), 0.0)),
        "mean_clip": float(coeffs_clip[c_idx]),
        "var_clip": float(max(np.dot(coeffs_clip[nc_mask], coeffs_clip[nc_mask]), 0.0)),
    })
df_conv = pd.DataFrame(conv_rows)

N_MC_ref = 1_000_000
rng_ref  = np.random.default_rng(seed=42)
f_raw_ref = (Pwf_cv + rng_ref.normal(0, sig_P, N_MC_ref)
             + (q_cv + rng_ref.normal(0, 1, N_MC_ref) * sig_q_cv) * rng_ref.uniform(eta_lo, eta_hi, N_MC_ref))
var_MC_noclip = float(np.var(f_raw_ref))
var_MC_clip   = float(np.var(np.clip(f_raw_ref, 0.0, Pi)))

df_conv["var_MC_noclip"]  = var_MC_noclip
df_conv["var_MC_clip"]    = var_MC_clip
df_conv["err_noclip_pct"] = df_conv["var_noclip"].apply(lambda v: abs(v - var_MC_noclip) / var_MC_noclip * 100)
df_conv["err_clip_pct"]   = df_conv["var_clip"].apply(lambda v: abs(v - var_MC_clip) / var_MC_clip * 100)
df_conv.to_csv(os.path.join(out_dir, "pce_convergence.csv"), index=False)

print(df_conv[["p", "terms", "err_noclip_pct", "err_clip_pct"]].to_string(index=False))

"""Coverage, error statistics, Sobol' summary"""

n_common = min(n_days, len(Pavg_true))
Pt   = Pavg_true[:n_common]
Pm   = Pavg_mean[:n_common]
meth = pce_method[:n_common]
valid = meth != "none"
n_valid = int(valid.sum())

err  = Pm[valid] - Pt[valid]
MAE  = float(np.mean(np.abs(err)))
RMSE = float(np.sqrt(np.mean(err ** 2)))
R2   = 1.0 - float(np.sum(err ** 2)) / float(np.sum((Pt[valid] - Pt[valid].mean()) ** 2))

cover_tot = 100.0 * float(np.mean((Pt[valid] >= Pavg_p025[:n_common][valid]) & (Pt[valid] <= Pavg_p975[:n_common][valid])))
width_tot = float(np.mean(Pavg_p975[:n_common][valid] - Pavg_p025[:n_common][valid]))
cover_epi = 100.0 * float(np.mean((Pt[valid] >= Pavg_p025_epi[:n_common][valid]) & (Pt[valid] <= Pavg_p975_epi[:n_common][valid])))
width_epi = float(np.mean(Pavg_p975_epi[:n_common][valid] - Pavg_p025_epi[:n_common][valid]))

print(f"MAE={MAE:.4f} psi  RMSE={RMSE:.4f} psi  R2={R2:.6f}")
print(f"Total CI95: cov={cover_tot:.2f}%  width={width_tot:.3f} psi")
print(f"Epistemic CI95: cov={cover_epi:.2f}%  width={width_epi:.3f} psi")

S1_v = Sobol_S1[:n_common][valid]
pi_mask_c  = valid & (meth == "PI")
mbe_mask_c = valid & (meth == "MBE")
S1_all = np.nanmean(S1_v, axis=0)
S1_pi  = np.nanmean(Sobol_S1[:n_common][pi_mask_c], axis=0)
S1_mbe = np.nanmean(Sobol_S1[:n_common][mbe_mask_c], axis=0)
ST_all = np.nanmean(Sobol_ST[:n_common][valid], axis=0)

print("Mean first-order Sobol' indices:")
for lbl, sa, sp, sm in zip(labels, S1_all, S1_pi, S1_mbe):
    print(f"  {lbl:22s}  all={sa:.4f}  PI={sp:.4f}  MBE={sm:.4f}")

"""Save UQ_results.csv and Sobol_indices.csv"""

dominant = [labels[int(np.nanargmax(Sobol_S1[i]))] if not np.all(np.isnan(Sobol_S1[i])) else "unknown"
            for i in range(n_days)]

pd.DataFrame({
    "day": np.arange(n_days), "pce_method": pce_method,
    "Pavg_mean_pce": Pavg_mean, "Pavg_std_pce": Pavg_std, "Pavg_var_pce": Pavg_var,
    "Pavg_p025_total": Pavg_p025, "Pavg_p975_total": Pavg_p975,
    "CI95_width_total": Pavg_p975 - Pavg_p025,
    "Pavg_p025_epi": Pavg_p025_epi, "Pavg_p975_epi": Pavg_p975_epi,
    "CI95_width_epi": Pavg_p975_epi - Pavg_p025_epi,
    "Sobol_S1_eP": Sobol_S1[:, 0], "Sobol_S1_eq": Sobol_S1[:, 1],
    "Sobol_S1_J": Sobol_S1[:, 2], "Sobol_S1_ct": Sobol_S1[:, 3],
    "Sobol_ST_eP": Sobol_ST[:, 0], "Sobol_ST_eq": Sobol_ST[:, 1],
    "Sobol_ST_J": Sobol_ST[:, 2], "Sobol_ST_ct": Sobol_ST[:, 3],
    "dominant_source": dominant,
    "Pavg_p050": Pavg_p050, "Pavg_p250": Pavg_p250, "Pavg_p500": Pavg_p500,
    "Pavg_p750": Pavg_p750, "Pavg_p950": Pavg_p950,
    "Pavg_skew": Pavg_skew, "Pavg_kurt": Pavg_kurt,
}).to_csv(os.path.join(out_dir, "UQ_results.csv"), index=False)

pd.DataFrame({"Input": labels, "S1_mean": S1_all, "ST_mean": ST_all}
             ).to_csv(os.path.join(out_dir, "Sobol_indices.csv"), index=False)

print("Saved: outputs/{UQ_results,Sobol_indices,pce_convergence}.csv")

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 300})

"""Figure 4: PCE convergence, clipped vs unclipped"""

orders_arr = df_conv["p"].values
err_nc, err_c = df_conv["err_noclip_pct"].values, df_conv["err_clip_pct"].values

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("PCE Convergence Study: Clipped vs Unclipped PI Forward Model", fontsize=10, fontweight="bold")

ax = axes[0]
ax.semilogy(orders_arr, err_nc, "o-", color="#2ca02c", lw=2, ms=6, label="Unclipped (bilinear)\nFloor = MC noise")
for o, e in zip(orders_arr, err_nc):
    ax.text(o, e * 1.8, f"{e:.4f}%", ha="center", fontsize=8)
ax.axhline(0.1, color="gray", ls=":", lw=0.8, label="0.1% reference")
ax.set(xlabel="PCE order p", ylabel="% error vs MC (N=10⁶)",
       title="(A) Unclipped model — converges exactly at p=2", xticks=orders_arr)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(orders_arr, err_c, "s-", color="#d62728", lw=2, ms=6, label="Clipped (Gibbs divergence)")
for o, e in zip(orders_arr, err_c):
    ax.text(o + 0.05, e + 0.3, f"{e:.2f}%", ha="left", fontsize=8)
ax.set(xlabel="PCE order p", ylabel="% error vs MC clip (N=10⁶)",
       title="(B) Clipped model — Gibbs phenomenon: diverges with p", xticks=orders_arr)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, f"MC reference\nvar_noclip = {var_MC_noclip:,.0f} psi²\nvar_clip = {var_MC_clip:,.0f} psi²",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig04_pce_convergence.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig04_pce_convergence.png")

"""Figure 6: PCE mean + dual CI vs ground truth, 3 stacked panels"""

t_d = np.arange(n_days)
mbe_days = np.where(pce_method == "MBE")[0]

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle(
    f"PCE Mean ± Dual CI vs True P̄  —  365-Day Digital Twin (Well 15/9-F-14 H)\n"
    f"MAE={MAE:.2f} psi  RMSE={RMSE:.2f} psi  R²={R2:.4f}  "
    f"Total-CI95 cov={cover_tot:.0f}%  Epi-CI95 cov={cover_epi:.0f}%",
    fontsize=10, fontweight="bold")

ax = axes[0]
ax.fill_between(t_d, Pavg_p025, Pavg_p975, alpha=0.18, color="#1f77b4",
                 label=f"Total CI₉₅ (ε+J+cₜ) — avg width {width_tot:.0f} psi — cov={cover_tot:.0f}%")
ax.fill_between(t_d, Pavg_p025_epi, Pavg_p975_epi, alpha=0.40, color="#2ca02c",
                 label=f"Epistemic CI₉₅ (J+cₜ) — avg width {width_epi:.0f} psi — cov={cover_epi:.0f}%")
ax.plot(t_d, Pavg_mean, color="#08306b", lw=1.8, label=f"PCE mean (MAE={MAE:.2f} psi, R²={R2:.4f})")
ax.plot(np.arange(len(Pavg_true)), Pavg_true, color="black", lw=1.5, label="True P̄ (ground truth)")
if len(mbe_days):
    ax.scatter(mbe_days, Pavg_mean[mbe_days], s=20, color="orange", zorder=5, label="MBE days (pressure buildup)")
ax.set(xlabel="Day", ylabel=r"$\bar{P}$ (psi)", title="(A) PCE Mean + Dual Confidence Intervals vs Ground Truth")
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

ax = axes[1]
resid = Pavg_mean - Pavg_true[:n_days]
ax.plot(t_d, resid, color="#1f77b4", lw=0.8, alpha=0.8, label="PCE mean − truth")
ax.axhline(MAE, ls="--", color="red", lw=1.0, label=f"±MAE = {MAE:.2f} psi")
ax.axhline(-MAE, ls="--", color="red", lw=1.0)
ax.axhline(0, color="black", lw=0.8)
ax.set(xlabel="Day", ylabel="Error (psi)", title=f"(B) PCE Mean Residuals  [RMSE={RMSE:.2f} psi  R²={R2:.4f}]")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(t_d, Pavg_std, color="#ff7f0e", lw=1.0)
ax.fill_between(t_d, 0, Pavg_std, alpha=0.25, color="#ff7f0e")
ax.set(xlabel="Day", ylabel=r"$\sigma_{\bar{P}}$ (psi)", title="(C) PCE Standard Deviation Over Time")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig06_pavg_timeseries.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig06_pavg_timeseries.png")

"""Figure 7: First-order Sobol' attribution map, 2 stacked panels"""

s1_eP = np.where(np.isnan(Sobol_S1[:, 0]), 0, Sobol_S1[:, 0])
s1_eq = np.where(np.isnan(Sobol_S1[:, 1]), 0, Sobol_S1[:, 1])
s1_J  = np.where(np.isnan(Sobol_S1[:, 2]), 0, Sobol_S1[:, 2])
s1_ct = np.where(np.isnan(Sobol_S1[:, 3]), 0, Sobol_S1[:, 3])
colors_s = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
fig.suptitle("First-Order Sobol' Attribution Map", fontsize=10, fontweight="bold")

ax = axes[0]
ax.stackplot(t_d, s1_eP, s1_eq, s1_J, s1_ct,
             labels=[r"$S_1^{\varepsilon_P}$", r"$S_1^{\varepsilon_q}$", r"$S_1^{J}$", r"$S_1^{c_t}$"],
             colors=colors_s, alpha=0.85)
for md in mbe_days:
    ax.axvline(md, color="gray", lw=0.5, alpha=0.5)
ax.set(xlabel="Day", ylabel="First-order Sobol' index S₁",
       title="(A) Stacked First-Order Sobol' Indices", ylim=(0, 1.05))
ax.legend(fontsize=9, ncol=4, loc="lower right"); ax.grid(True, alpha=0.2)

ax = axes[1]
pi_mask_plot = pce_method == "PI"
ax.bar(t_d[pi_mask_plot], Sobol_S1[pi_mask_plot, 2], width=0.8, color="#1f77b4", alpha=0.7,
       label=f"S₁ J (PI days, mean={np.nanmean(Sobol_S1[pi_mask_plot, 2]):.4f})")
ax.bar(t_d[mbe_days], Sobol_S1[mbe_days, 3], width=0.8, color="#2ca02c", alpha=0.9,
       label="S₁ cₜ (MBE days, mean=1.000)")
ax.set(xlabel="Day", ylabel="First-order Sobol' S₁", title="(B) Dominant Sobol' Index Per Day", ylim=(0, 1.1))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig07_sobol_attribution.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig07_sobol_attribution.png")

"""Figure 8: mean Sobol' by mode + S1 vs ST"""

x4, w4 = np.arange(4), 0.22
labels_s = [r"$\varepsilon_P$ (BHP noise)", r"$\varepsilon_q$ (MPFM noise)", r"$J$ (PI uncert.)", r"$c_t$ (compress.)"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Mean First-Order Sobol' Sensitivity Indices", fontsize=10, fontweight="bold")

ax = axes[0]
b1 = ax.bar(x4 - w4, S1_all, w4, label="All days", color="#1f77b4", edgecolor="k", lw=0.5)
b2 = ax.bar(x4,      S1_pi,  w4, label="PI days",  color="#2ca02c", edgecolor="k", lw=0.5)
b3 = ax.bar(x4 + w4, S1_mbe, w4, label="MBE days", color="#ff7f0e", edgecolor="k", lw=0.5)
for bars in (b1, b2, b3):
    for b in bars:
        h = b.get_height()
        if h > 0.01:
            ax.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=7)
ax.set(xticks=x4, xticklabels=labels_s, ylabel=r"Mean $S_1$", title="(A) First-Order Sobol' by Mode", ylim=(0, 1.15))
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
# S1 and ST must come from the same day-subset (S_T >= S1 always holds otherwise)
ax.bar(x4 - 0.15, S1_all, 0.28, label=r"$S_1$ (all)", color="#2ca02c", edgecolor="k", lw=0.5)
ax.bar(x4 + 0.15, ST_all, 0.28, label=r"$S_T$ (all)", color="#9467bd", edgecolor="k", lw=0.5, alpha=0.7)
ax.set(xticks=x4, xticklabels=labels_s, ylabel="Sobol' index",
       title=r"(B) $S_1$ vs $S_T$",
       ylim=(0, 1.15))
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig08_sobol_summary.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig08_sobol_summary.png")

"""Figure 1: orthonormal polynomial basis functions (Hermite, Legendre)"""

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Orthogonal Polynomial Basis Functions (Wiener-Askey Scheme)", fontsize=10, fontweight="bold")
col4 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

x_h, w_h = np.linspace(-4, 4, 400), None
w_h = stats.norm.pdf(x_h)
ax = axes[0]
for n in range(4):
    c = np.zeros(n + 1); c[-1] = 1.0
    vals = hermeval(x_h, c) / np.sqrt(math.factorial(n))
    ax.plot(x_h, vals, color=col4[n], lw=1.8, label=rf"$He_{n}(x)/\sqrt{{{n}!}}$")
ax.fill_between(x_h, -0.6 * w_h / w_h.max(), 0, color="grey", alpha=0.15, label="w(x) Gaussian")
ax.axhline(0, color="black", lw=0.5)
ax.set(xlabel="x", ylabel=r"Normalised $He_n(x)$",
       title=r"(A) Probabilists' Hermite — for $\varepsilon_P$, $\varepsilon_q$", xlim=(-4, 4), ylim=(-1.6, 1.6))
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

x_l = np.linspace(-1, 1, 400)
ax = axes[1]
for n in range(4):
    c = np.zeros(n + 1); c[-1] = 1.0
    vals = legval(x_l, c) / np.sqrt(2.0 / (2 * n + 1))
    ax.plot(x_l, vals, color=col4[n], lw=1.8, label=rf"$P_{n}(x)/\sqrt{{2/(2\times{n}+1)}}$")
ax.axhspan(-0.62, -0.55, color="grey", alpha=0.15)
ax.text(0, -0.585, "w(x)=1/2 Uniform", ha="center", fontsize=8, color="grey")
ax.axhline(0, color="black", lw=0.5)
ax.set(xlabel="x", ylabel=r"Normalised $P_n(x)$",
       title=r"(B) Legendre — for $J$ (via $\eta=1/J$) and $c_t$ (via $\zeta=1/c_t$)", xlim=(-1, 1), ylim=(-0.65, 1.6))
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig01_basis_functions.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig01_basis_functions.png")

"""Figure 2: uncertain-input probability distributions (Askey scheme)"""

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
fig.suptitle("Uncertain Inputs: Wiener-Askey Probability Distributions", fontsize=10, fontweight="bold")

x = np.linspace(-6, 6, 400)
ax = axes[0]
ax.plot(x, stats.norm.pdf(x, 0, sig_P), color=colors_s[0], lw=2)
ax.fill_between(x, stats.norm.pdf(x, 0, sig_P), alpha=0.25, color=colors_s[0])
ax.axvline(0, ls="--", color="gray", lw=0.8)
ax.set(xlabel=r"$\varepsilon_P$ (psi)", ylabel="PDF",
       title=r"$\varepsilon_P \sim \mathcal{N}(0,\,4\,\mathrm{psi}^2)$" + f"\n(σ={sig_P} psi)")
ax.grid(True, alpha=0.3)

q_ref_plt = 15000.0
sig_q_ref_plt = sig_q_frac * q_ref_plt
x2 = np.linspace(-3 * sig_q_ref_plt, 3 * sig_q_ref_plt, 400)
ax = axes[1]
ax.plot(x2, stats.norm.pdf(x2, 0, sig_q_ref_plt), color=colors_s[1], lw=2)
ax.fill_between(x2, stats.norm.pdf(x2, 0, sig_q_ref_plt), alpha=0.25, color=colors_s[1])
ax.axvline(0, ls="--", color="gray", lw=0.8)
ax.set(xlabel=r"$\varepsilon_q$ (STB/day)", ylabel="PDF",
       title=r"$\varepsilon_q \sim \mathcal{N}(0,\,(0.02q)^2)$" + f"\n(σ=2%×q; shown at q={q_ref_plt:,.0f})")
ax.grid(True, alpha=0.3)

x3 = np.linspace(J_lo * 0.95, J_hi * 1.05, 400)
pdf3 = np.where((x3 >= J_lo) & (x3 <= J_hi), 1.0 / (J_hi - J_lo), 0.0)
ax = axes[2]
ax.plot(x3, pdf3, color=colors_s[2], lw=2)
ax.fill_between(x3, pdf3, alpha=0.25, color=colors_s[2])
ax.axvline(J0, ls="--", color="gray", lw=0.8, label=f"J₀={J0:.2f}")
ax.set(xlabel="J (STB/day/psi)", ylabel="PDF",
       title=r"$J \sim \mathcal{U}(\pm 15\%)$" + f"\n[{J_lo:.3f}, {J_hi:.3f}]")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

x4p = np.linspace(ct_lo * 0.9, ct_hi * 1.1, 400)
pdf4 = np.where((x4p >= ct_lo) & (x4p <= ct_hi), 1.0 / (ct_hi - ct_lo), 0.0)
ax = axes[3]
ax.plot(x4p * 1e6, pdf4 / 1e6, color=colors_s[3], lw=2)
ax.fill_between(x4p * 1e6, pdf4 / 1e6, alpha=0.25, color=colors_s[3])
ax.axvline(ct0 * 1e6, ls="--", color="gray", lw=0.8, label=f"ct₀={ct0 * 1e6:.0f}×10⁻⁶")
ax.set(xlabel=r"$c_t$ (×10⁻⁶ psi⁻¹)", ylabel="PDF",
       title=r"$c_t \sim \mathcal{U}(\pm 20\%)$" + f"\n[{ct_lo * 1e6:.0f}, {ct_hi * 1e6:.0f}]×10⁻⁶")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig02_input_distributions.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig02_input_distributions.png")

"""Figure 3: Smolyak sparse grid vs full tensor product"""
n_full_5pt = 5 ** 4
pct_reduction = (1 - n_nodes / n_full_5pt) * 100

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle(r"Smolyak Sparse Grid vs Full Tensor Product — $d=4$, $\ell=3$", fontsize=10, fontweight="bold")

ax = axes[0]
ax.scatter(nodes[0], nodes[1], s=20, color="#1f77b4", alpha=0.7, edgecolors="k", lw=0.3)
ax.set(xlabel=r"$\xi_1$ ($\varepsilon_P$, normalised)", ylabel=r"$\xi_2$ ($\varepsilon_q$, normalised)",
       title=f"(A) Smolyak Sparse Grid\n{n_nodes} nodes ($\\varepsilon_P$–$\\varepsilon_q$ projection)")
ax.grid(True, alpha=0.3)

grid_1d = np.linspace(-3, 3, 5)
gx, gy = np.meshgrid(grid_1d, grid_1d)
ax = axes[1]
ax.scatter(gx.ravel(), gy.ravel(), s=20, color="#d62728", alpha=0.7, edgecolors="k", lw=0.3)
ax.set(xlabel=r"$\xi_1$", ylabel=r"$\xi_2$",
       title=f"(B) Full Tensor Product (5-pt Gauss-Hermite)\n"
             f"5²=25 shown (5⁴={n_full_5pt} in d=4); Smolyak saves {pct_reduction:.0f}%")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig03_smolyak_grid.png"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: outputs/fig03_smolyak_grid.png  ({n_nodes} Smolyak vs {n_full_5pt} full tensor, {pct_reduction:.0f}% reduction)")

"""Figure 9: PCE mean vs true Pavg scatter + residual histogram"""

pred_v, true_v, err_v = Pm[valid], Pt[valid], err
n_pi_v, n_mbe_v = int((meth[valid] == "PI").sum()), int((meth[valid] == "MBE").sum())

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle(f"PCE Surrogate Validation — {n_valid} Valid Days", fontsize=10, fontweight="bold")

ax = axes[0]
ax.scatter(true_v, pred_v, s=8, alpha=0.6, color="#1f77b4", edgecolors="none")
lims = [min(true_v.min(), pred_v.min()) - 50, max(true_v.max(), pred_v.max()) + 50]
ax.plot(lims, lims, "k--", lw=1.0, label="Perfect fit")
ax.set(xlabel=r"True $\bar{P}$ (psi)", ylabel=r"PCE Mean $\hat{P}$ (psi)",
       title=f"(A) PCE vs True P̄\nR²={R2:.4f}  MAE={MAE:.2f} psi", xlim=lims, ylim=lims, aspect="equal")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.text(0.05, 0.92, f"n={n_valid}  PI={n_pi_v}  MBE={n_mbe_v}", transform=ax.transAxes, fontsize=8)

ax = axes[1]
ax.hist(err_v, bins=40, color="#1f77b4", edgecolor="white", lw=0.3, density=True, alpha=0.8)
x_fit = np.linspace(err_v.min(), err_v.max(), 300)
ax.plot(x_fit, stats.norm.pdf(x_fit, err_v.mean(), err_v.std()), "r-", lw=1.5, label="Gaussian fit")
ax.axvline(0, color="black", lw=1.0)
ax.axvline(err_v.mean(), ls="--", color="red", lw=1.0, label=f"Mean={err_v.mean():.2f} psi")
ax.set(xlabel="Residual (psi)", ylabel="Density",
       title=f"(B) Residual Distribution\nRMSE={RMSE:.2f} psi  skew={float(stats.skew(err_v)):.3f}")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig09_residuals.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig09_residuals.png")

"""Figure 10: standardized-residual diagnostic (PI days only)"""

pi_mask_z = meth == "PI"
z = (Pm[pi_mask_z] - Pt[pi_mask_z]) / Pavg_std[:n_common][pi_mask_z]
z_mean, z_std = float(np.mean(z)), float(np.std(z))
ks_stat, ks_p = stats.kstest(z, "norm")
n_pi_z = len(z)

pd.DataFrame({
    "day": np.where(pi_mask_z)[0], "P_true": Pt[pi_mask_z], "P_pce": Pm[pi_mask_z],
    "sigma_pce": Pavg_std[:n_common][pi_mask_z], "z": z,
}).to_csv(os.path.join(out_dir, "standardised_residuals.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle(
    r"Standardised Residuals: $z_i = (\bar{P}_{\mathrm{PCE},i} - \bar{P}_{\mathrm{true},i}) / \sigma_{\mathrm{PCE},i}$"
    f"\nDigital twin validation — {n_pi_z} stable PI days  |  "
    rf"$\bar{{z}}={z_mean:.4f}$,  $s_z={z_std:.4f}$,  K-S vs $\mathcal{{N}}(0,1)$: {ks_stat:.4f}",
    fontsize=10)

ax = axes[0]
z_sorted = np.sort(z)
ecdf = (np.arange(1, len(z_sorted) + 1) - 0.5) / len(z_sorted)
z_grid = np.linspace(-4, 4, 400)
ax.plot(z_sorted, ecdf, color="#2980b9", lw=1.5, label="Empirical CDF of $z$")
ax.plot(z_grid, stats.norm.cdf(z_grid), "k--", lw=1.5, label=r"$\mathcal{N}(0,1)$ CDF")
ax.axvline(0, color="#aaa", lw=0.8, ls=":")
ax.set(xlabel="$z_i$", ylabel="Cumulative probability", title=r"Empirical CDF vs $\mathcal{N}(0,1)$", xlim=(-4, 4))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.annotate(f"$\\bar{{z}}={z_mean:.4f}$\n$s_z={z_std:.4f}$\nK-S={ks_stat:.4f}",
            xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#aaa", alpha=0.8))

ax = axes[1]
ax.hist(z, bins=30, density=True, color="#3498db", alpha=0.6, label="$z_i$ histogram")
ax.plot(z_grid, stats.norm.pdf(z_grid), "k--", lw=1.5, label=r"$\mathcal{N}(0,1)$")
ax.plot(z_grid, stats.norm.pdf(z_grid, z_mean, z_std), "r-", lw=1.5,
        label=f"$\\mathcal{{N}}({z_mean:.3f},{z_std:.3f}^2)$ fitted")
ax.set(xlabel="$z_i$", ylabel="Density", title="Distribution of $z_i$")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig10_standardised_residuals.png"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: outputs/fig10_standardised_residuals.png  (z_mean={z_mean:.4f} z_std={z_std:.4f} KS={ks_stat:.4f})")

"""Figure 12: PCE vs direct Monte Carlo, reference day (highest-variance PI day)"""

pi_std = np.where(pce_method == "PI", Pavg_std, -np.inf)
ref_day_mc = int(np.nanargmax(pi_std))
Pwf_ref287 = float(P_arr[ref_day_mc])
q_ref287   = float(q_arr[ref_day_mc])
sig_q_287  = sig_q_frac * q_ref287
print(f"MC-vs-PCE reference day: {ref_day_mc}  (Pwf={Pwf_ref287:.1f} psi, q={q_ref287:.0f} STB/day, "
      f"sigma_PCE={Pavg_std[ref_day_mc]:.1f} psi)")

N_MC_11 = 500_000
rng11 = np.random.default_rng(seed=42)
eps_P_11 = rng11.normal(0, sig_P, N_MC_11)
eps_q_11 = rng11.normal(0, sig_q_287, N_MC_11)
eta_11   = rng11.uniform(eta_lo, eta_hi, N_MC_11)
mc_raw_11  = Pwf_ref287 + eps_P_11 + (q_ref287 + eps_q_11) * eta_11
mc_clip_11 = np.clip(mc_raw_11, 0.0, Pi)

eps_q_n_11 = nodes[1] * sig_q_287
model_evals_11 = Pwf_ref287 + nodes[0] + (q_ref287 + eps_q_n_11) * nodes[2]
coeffs_11 = poly_at_nodes_w @ model_evals_11

mc_inputs_11 = joint.sample(N_MC_11, rule="random", seed=42)
poly_at_mc_11 = np.array(poly(*mc_inputs_11), dtype=float)
pce_clip_11 = np.clip(coeffs_11 @ poly_at_mc_11, 0.0, Pi)

ks_11, _ = stats.ks_2samp(mc_clip_11, pce_clip_11)
d_mean_11 = abs(float(np.mean(mc_clip_11)) - float(np.mean(pce_clip_11)))
d_std_11  = abs(float(np.std(mc_clip_11)) - float(np.std(pce_clip_11)))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle(f"PCE vs Direct MC Validation — Reference Day {ref_day_mc}\n"
             f"(Pwf={Pwf_ref287:.1f} psi, q={q_ref287:.0f} STB/day, σ_q={sig_q_287:.0f} STB/day)",
             fontsize=10, fontweight="bold")

ax = axes[0]
bins11 = np.linspace(min(mc_clip_11.min(), pce_clip_11.min()), Pi, 120)
ax.hist(mc_clip_11,  bins=bins11, density=True, alpha=0.5, color="#1f77b4", label=f"Direct MC (N={N_MC_11:,})")
ax.hist(pce_clip_11, bins=bins11, density=True, alpha=0.5, color="#d62728", label="PCE surrogate")
ax.axvline(float(np.mean(pce_clip_11)), ls="--", color="#08306b", lw=1.5,
           label=f"PCE mean={np.mean(pce_clip_11):.1f} psi")
ax.axvline(float(np.mean(mc_clip_11)), ls=":", color="#1f77b4", lw=1.5,
           label=f"MC mean={np.mean(mc_clip_11):.1f} psi")
ax.set(xlabel=r"$\bar{P}$ (psi)", ylabel="Density",
       title=f"(A) PDF Overlay\nKS stat={ks_11:.4f}  |Δmean|={d_mean_11:.2f} psi")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
sorted_mc, sorted_pce = np.sort(mc_clip_11), np.sort(pce_clip_11)
cdf_p_11 = np.arange(1, N_MC_11 + 1) / N_MC_11
ax.plot(sorted_mc,  cdf_p_11, color="#1f77b4", lw=1.5, label="Direct MC CDF")
ax.plot(sorted_pce, cdf_p_11, color="#d62728", lw=1.5, ls="--", label="PCE surrogate CDF")
ax.set(xlabel=r"$\bar{P}$ (psi)", ylabel="CDF", title=f"(B) CDF Comparison\n|Δσ|={d_std_11:.2f} psi")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig12_mc_vs_pce.png"), bbox_inches="tight")
plt.close(fig)
print(f"Saved: outputs/fig12_mc_vs_pce.png  (|dmean|={d_mean_11:.2f} psi  |dstd|={d_std_11:.2f} psi)")

"""Figure 11: prior-shift stress test (non-circularity check)"""

J_lo_sh, J_hi_sh     = J0 * 1.10, J0 * 1.40
eta_lo_sh, eta_hi_sh = 1.0 / J_hi_sh, 1.0 / J_lo_sh

joint_sh = cp.J(cp.Normal(0, sig_P), cp.Normal(0, 1.0),
                cp.Uniform(eta_lo_sh, eta_hi_sh), cp.Uniform(zeta_lo, zeta_hi))
poly_sh             = cp.generate_expansion(PCE_ORDER, joint_sh, normed=True)
nodes_sh, weights_sh = cp.generate_quadrature(SPARSE_LEVEL, joint_sh, rule="gaussian", sparse=True)
poly_at_nodes_sh    = np.array(poly_sh(*nodes_sh), dtype=float)
poly_at_nodes_w_sh  = poly_at_nodes_sh * weights_sh[np.newaxis, :]
mc_sh               = joint_sh.sample(N_MC, rule="random", seed=42)
poly_at_mc_sh       = np.array(poly_sh(*mc_sh), dtype=float)

Pm_sh   = np.full(n_days, np.nan)
p025_sh = np.full(n_days, np.nan)
p975_sh = np.full(n_days, np.nan)
meth_sh = np.full(n_days, "none", dtype=object)

for i in range(n_days):
    s = st_arr[i]
    try:
        if s == 0:
            Pwf_t, q_t = float(P_arr[i]), float(q_arr[i])
            if not (np.isfinite(Pwf_t) and np.isfinite(q_t) and q_t > 0):
                continue
            sig_q_t = sig_q_frac * q_t
            model_evals = Pwf_t + nodes_sh[0] + (q_t + nodes_sh[1] * sig_q_t) * nodes_sh[2]
            coeffs = poly_at_nodes_w_sh @ model_evals
            pce_mc = np.clip(coeffs @ poly_at_mc_sh, 0.0, Pi)
            meth_sh[i] = "PI"
        elif s == 2:
            Np_t = float(Np_arr[i])
            if not np.isfinite(Np_t):
                continue
            model_evals = Pi - Np_t * nodes_sh[3] / N_ooip
            coeffs = poly_at_nodes_w_sh @ model_evals
            pce_mc = np.clip(coeffs @ poly_at_mc_sh, 0.0, Pi)
            meth_sh[i] = "MBE"
        else:
            continue
        Pm_sh[i] = np.mean(pce_mc)
        p025_sh[i], p975_sh[i] = np.percentile(pce_mc, [2.5, 97.5])
    except Exception:
        pass

pi_sh    = meth_sh[:n_common] == "PI"
err_sh   = Pm_sh[:n_common][pi_sh] - Pt[pi_sh]
MAE_sh   = float(np.mean(np.abs(err_sh)))
RMSE_sh  = float(np.sqrt(np.mean(err_sh ** 2)))
covers_sh = (p025_sh[:n_common][pi_sh] <= Pt[pi_sh]) & (Pt[pi_sh] <= p975_sh[:n_common][pi_sh])
cov_sh    = float(np.mean(covers_sh) * 100)

pd.DataFrame({
    "day": np.where(pi_sh)[0], "Pavg_true": Pt[pi_sh], "Pavg_pce_shift": Pm_sh[:n_common][pi_sh],
    "ci_lo": p025_sh[:n_common][pi_sh], "ci_hi": p975_sh[:n_common][pi_sh], "covered": covers_sh.astype(int),
}).to_csv(os.path.join(out_dir, "prior_stress_test.csv"), index=False)

print(f"Prior stress test: J shifted to [{J_lo_sh:.3f}, {J_hi_sh:.3f}] (excludes true J0={J0:.3f})")
print(f"  Shifted MAE={MAE_sh:.2f} psi (vs original {MAE:.2f} psi)")
print(f"  Shifted 95% CI coverage={cov_sh:.2f}% (vs original {cover_tot:.2f}%)")
print("Saved: outputs/prior_stress_test.csv")

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(t_d[pi_sh], p025_sh[:n_common][pi_sh], p975_sh[:n_common][pi_sh],
                 alpha=0.20, color="#1f77b4", label=f"95% CI, shifted prior — cov={cov_sh:.0f}%")
ax.plot(t_d[pi_sh], Pm_sh[:n_common][pi_sh], color="#1f77b4", lw=1.6,
        label=f"PCE mean, shifted prior (MAE={MAE_sh:.0f} psi)")
ax.plot(t_d[pi_sh], Pt[pi_sh], color="black", lw=2.0, ls="--", label="True P̄")
ax.set(xlabel="Day (stable-PI days only)", ylabel=r"$\bar{P}$ (psi)",
       title="Prior-Shift Stress Test: 95% CI vs Ground Truth")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig11_prior_stress_test.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig11_prior_stress_test.png")
