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
Pi     = float(params["Pi"])
J0_csv = float(params["J0"])   

ct0        = 15.0e-6
sig_P      = 2.0
sig_q_frac = 0.02

ct_lo, ct_hi     = ct0 * 0.80, ct0 * 1.20
zeta_lo, zeta_hi = 1.0 / ct_hi, 1.0 / ct_lo

PCE_ORDER    = 3
SPARSE_LEVEL = 3
N_MC         = 10_000
labels = ["ep_P (BHP noise)", "ep_q (flow noise)", "J (PI uncert.)", "ct (compress.)"]

J_DRIFT_THR  = 0.15   # 15% deviation from J_base triggers drift flag
RECALIB_WIN  = 48     # 48-calendar-day lookback
DRAWDOWN_MIN = 100.0  # psi, minimum drawdown for J estimation
MIN_SAMPLES  = 10      # minimum pairs required to recalibrate

print(f"Pi={Pi:.1f} psi  J0_csv(ref)={J0_csv:.4f} STB/d/psi  ct0={ct0:.1e} psi^-1")

"""Load real Volve F-14 daily record (3,056 days), assign states, N_ooip and J_base"""

df    = pd.read_csv(os.path.join(out_dir, "Volve_F14_Pavg_daily.csv"), parse_dates=["date"])
n_days = len(df)
dates  = pd.to_datetime(df["date"].values)

q_arr    = df["q_oil_STB_day"].values.astype(float)
P_arr    = df["BHP_psi"].values.astype(float)
Pavg_ref = df["Pavg_MBE_psi"].values.astype(float)
st_arr   = np.where(df["producing_day"].values == 1, 0, np.where(df["shutin_day"].values == 1, 2, -1))

Np_arr     = np.cumsum(q_arr)
Pavg_final = float(np.nanmin(Pavg_ref))
N_ooip     = float(Np_arr[-1]) / (ct0 * (Pi - Pavg_final))

# J_base
dd_all    = Pavg_ref - P_arr
base_mask = (st_arr == 0) & (dd_all > DRAWDOWN_MIN) & np.isfinite(dd_all)
J_base    = float(np.median(q_arr[base_mask] / dd_all[base_mask]))

print(f"Loaded {n_days} days ({dates[0].date()} to {dates[-1].date()})")
print(f"N_ooip={N_ooip/1e6:.2f} MMSTB  Pavg_final={Pavg_final:.1f} psi")
print(f"States: PI={(st_arr==0).sum()}  MBE={(st_arr==2).sum()}  skip={(st_arr==-1).sum()}")
print(f"J_base={J_base:.3f} STB/d/psi  (J0_csv={J0_csv:.3f}, ratio={J_base/J0_csv:.1f}x)")

"""PCE infrastructure: standardized d=4 joint"""

def sobol_masks(poly_arr, n_inputs):
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
    cp.Normal(0, sig_P),   # eps_P    [0]
    cp.Normal(0, 1.0),     # eps_q    [1] 
    cp.Uniform(-1, 1),     # u_eta    [2] 
    cp.Uniform(-1, 1),     # u_zeta   [3] 
)
poly           = cp.generate_expansion(PCE_ORDER, joint, normed=True)
nodes, weights = cp.generate_quadrature(SPARSE_LEVEL, joint, rule="gaussian", sparse=True)
n_nodes, n_terms = nodes.shape[1], len(poly)
assert abs(float(np.sum(weights)) - 1.0) < 1e-10

mc              = joint.sample(N_MC, rule="random", seed=42)
poly_at_nodes   = np.array(poly(*nodes), dtype=float)
poly_at_nodes_w = poly_at_nodes * weights[np.newaxis, :]
poly_at_mc      = np.array(poly(*mc), dtype=float)

mc_epi         = np.vstack([np.zeros((2, N_MC)), mc[2:4, :]])
poly_at_mc_epi = np.array(poly(*mc_epi), dtype=float)

S1_mask, ST_mask, is_const = sobol_masks(poly, 4)

# Fixed zeta physical values from standardized nodes 
zeta_phys_nodes = zeta_lo + (nodes[3] + 1.0) / 2.0 * (zeta_hi - zeta_lo)

print(f"Standardized joint: terms={n_terms}, nodes={n_nodes}")

"""Rolling J recalibration: 48-day lookback using Pavg_MBE as reference"""

J_eff_arr  = np.full(n_days, J_base)
drift_flag = np.zeros(n_days, dtype=bool)
drift_events = []

J_prev = J_base
for i in range(n_days):
    if st_arr[i] != 0:
        J_eff_arr[i] = J_prev
        continue

    t_lo  = dates[i] - pd.Timedelta(days=RECALIB_WIN)
    win   = (dates >= t_lo) & (dates <= dates[i]) & (st_arr == 0)
    win_dd, win_q = dd_all[win], q_arr[win]
    valid = (win_dd > DRAWDOWN_MIN) & np.isfinite(win_q) & np.isfinite(win_dd)

    J_eff_i = float(np.median(win_q[valid] / win_dd[valid])) if valid.sum() >= MIN_SAMPLES else J_prev
    J_eff_arr[i] = J_eff_i
    J_prev = J_eff_i

    if abs(J_eff_i - J_base) / J_base > J_DRIFT_THR:
        drift_flag[i] = True
        drift_events.append({"day_idx": i, "date": str(dates[i].date()),
                              "J_eff": J_eff_i, "J_base": J_base,
                              "drift_pct": (J_eff_i - J_base) / J_base * 100})

print(f"J_eff range (PI days): [{J_eff_arr[st_arr==0].min():.2f}, {J_eff_arr[st_arr==0].max():.2f}] STB/d/psi")
print(f"Drift events (|J_eff-J_base|/J_base > {J_DRIFT_THR*100:.0f}%): {len(drift_events)}")

"""Main PCE loop: per-day J_eff remapping (PI), fixed zeta (MBE)"""

Pavg_mean, Pavg_var, Pavg_std = (np.full(n_days, np.nan) for _ in range(3))
Pavg_p025, Pavg_p975         = np.full(n_days, np.nan), np.full(n_days, np.nan)
Pavg_p025_epi, Pavg_p975_epi = np.full(n_days, np.nan), np.full(n_days, np.nan)
Sobol_S1 = np.full((n_days, 4), np.nan)
Sobol_ST = np.full((n_days, 4), np.nan)
pce_method = np.full(n_days, "none", dtype=object)

for i in range(n_days):
    s = st_arr[i]
    try:
        if s == 0:
            Pwf_t, q_t = float(P_arr[i]), float(q_arr[i])
            if not (np.isfinite(Pwf_t) and np.isfinite(q_t) and q_t > 0):
                continue
            sig_q_t = sig_q_frac * q_t

            J_eff_i    = J_eff_arr[i]
            eta_eff_lo = 1.0 / (J_eff_i * 1.15)
            eta_eff_hi = 1.0 / (J_eff_i * 0.85)
            eta_n      = eta_eff_lo + (nodes[2] + 1.0) / 2.0 * (eta_eff_hi - eta_eff_lo)
            eps_q_n    = nodes[1] * sig_q_t

            model_evals = Pwf_t + nodes[0] + (q_t + eps_q_n) * eta_n
            coeffs = poly_at_nodes_w @ model_evals

            pce_mc     = np.clip(coeffs @ poly_at_mc,     0.0, Pi)
            pce_mc_epi = np.clip(coeffs @ poly_at_mc_epi, 0.0, Pi)
            pce_method[i] = "PI"

        elif s == 2:
            Np_t = float(Np_arr[i])
            if not np.isfinite(Np_t):
                continue
            model_evals = Pi - Np_t * zeta_phys_nodes / N_ooip
            coeffs = poly_at_nodes_w @ model_evals
            pce_mc     = np.clip(coeffs @ poly_at_mc, 0.0, Pi)
            pce_mc_epi = pce_mc   # no sensor noise in MBE -> total CI = epistemic CI
            pce_method[i] = "MBE"
        else:
            continue

        Pavg_mean[i] = np.mean(pce_mc)
        Pavg_var[i]  = np.var(pce_mc)
        Pavg_std[i]  = np.sqrt(Pavg_var[i])
        Pavg_p025[i], Pavg_p975[i]         = np.percentile(pce_mc,     [2.5, 97.5])
        Pavg_p025_epi[i], Pavg_p975_epi[i] = np.percentile(pce_mc_epi, [2.5, 97.5])

        var_pce = float(np.dot(coeffs[~is_const], coeffs[~is_const]))
        if var_pce > 1e-12:
            c2 = coeffs ** 2
            Sobol_S1[i] = np.clip(S1_mask @ c2 / var_pce, 0.0, 1.0)
            Sobol_ST[i] = np.clip(ST_mask @ c2 / var_pce, 0.0, 1.0)
    except Exception:
        pass

    if (i + 1) % 500 == 0 or i == n_days - 1:
        print(f"  Day {i+1:4d}/{n_days}  [{pce_method[i]}]  Pavg={Pavg_mean[i]:.1f} psi")

n_pi, n_mbe, n_none = (pce_method == "PI").sum(), (pce_method == "MBE").sum(), (pce_method == "none").sum()
print(f"PI={n_pi}  MBE={n_mbe}  Interpolated(skip)={n_none}")

"""Error statistics, coverage, and Sobol' summary"""

valid = pce_method != "none"
ref_v, est_v = Pavg_ref[valid], Pavg_mean[valid]
err = est_v - ref_v
MAE  = float(np.mean(np.abs(err)))
RMSE = float(np.sqrt(np.mean(err ** 2)))
R2   = 1.0 - float(np.sum(err ** 2)) / float(np.sum((ref_v - ref_v.mean()) ** 2))

cover_tot = 100.0 * float(np.mean((ref_v >= Pavg_p025[valid]) & (ref_v <= Pavg_p975[valid])))
width_tot = float(np.mean(Pavg_p975[valid] - Pavg_p025[valid]))
cover_epi = 100.0 * float(np.mean((ref_v >= Pavg_p025_epi[valid]) & (ref_v <= Pavg_p975_epi[valid])))
width_epi = float(np.mean(Pavg_p975_epi[valid] - Pavg_p025_epi[valid]))

print(f"MAE={MAE:.2f} psi  RMSE={RMSE:.2f} psi  R2={R2:.4f}")
print(f"Total CI95: cov={cover_tot:.2f}%  width={width_tot:.2f} psi")
print(f"Epistemic CI95: cov={cover_epi:.2f}%  width={width_epi:.2f} psi")

S1_out = np.nanmean(Sobol_S1[valid], axis=0)
ST_out = np.nanmean(Sobol_ST[valid], axis=0)
print("Mean first-order Sobol' indices:")
for lbl, v in zip(labels, S1_out):
    print(f"  {lbl:22s}: {v:.4f}")

"""Save UQ_results_real.csv, Sobol_indices_real.csv, and drift_events_real.csv"""

def interp_nan(arr):
    nans = np.isnan(arr)
    if 0 < nans.sum() < len(arr):
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr

for col in (Pavg_mean, Pavg_std, Pavg_var, Pavg_p025, Pavg_p975, Pavg_p025_epi, Pavg_p975_epi):
    interp_nan(col)

dominant = [labels[int(np.nanargmax(Sobol_S1[i]))] if not np.all(np.isnan(Sobol_S1[i])) else "unknown"
            for i in range(n_days)]

df_uq = pd.DataFrame({
    "day": np.arange(n_days), "date": dates, "pce_method": pce_method,
    "J_eff": J_eff_arr, "drift_flag": drift_flag.astype(int),
    "Pavg_mean_pce": Pavg_mean, "Pavg_std_pce": Pavg_std, "Pavg_var_pce": Pavg_var,
    "Pavg_ref_MBE": Pavg_ref,
    "Pavg_p025_total": Pavg_p025, "Pavg_p975_total": Pavg_p975,
    "CI95_width_total": Pavg_p975 - Pavg_p025,
    "Pavg_p025_epi": Pavg_p025_epi, "Pavg_p975_epi": Pavg_p975_epi,
    "CI95_width_epi": Pavg_p975_epi - Pavg_p025_epi,
    "Sobol_S1_eP": Sobol_S1[:, 0], "Sobol_S1_eq": Sobol_S1[:, 1],
    "Sobol_S1_J": Sobol_S1[:, 2], "Sobol_S1_ct": Sobol_S1[:, 3],
    "Sobol_ST_eP": Sobol_ST[:, 0], "Sobol_ST_eq": Sobol_ST[:, 1],
    "Sobol_ST_J": Sobol_ST[:, 2], "Sobol_ST_ct": Sobol_ST[:, 3],
    "dominant_source": dominant,
})
df_uq.to_csv(os.path.join(out_dir, "UQ_results_real.csv"), index=False)

pd.DataFrame({"Input": labels, "S1_mean": S1_out, "ST_mean": ST_out}
             ).to_csv(os.path.join(out_dir, "Sobol_indices_real.csv"), index=False)

df_drift = pd.DataFrame(drift_events) if drift_events else pd.DataFrame(
    columns=["day_idx", "date", "J_eff", "J_base", "drift_pct"])
df_drift.to_csv(os.path.join(out_dir, "drift_events_real.csv"), index=False)

print("Saved: outputs/{UQ_results_real,Sobol_indices_real,drift_events_real}.csv")

"""Diagnostics: inter-model divergence vs intra-method uncertainty, drift J_eff"""

pi_v = df_uq[df_uq["pce_method"] == "PI"]
diff_abs   = np.abs(pi_v["Pavg_mean_pce"].values - pi_v["Pavg_ref_MBE"].values)
sigma_pce_v = pi_v["Pavg_std_pce"].values
pcts = [5, 25, 50, 75, 95]
diff_pcts  = np.percentile(diff_abs, pcts)
sigma_pcts = np.percentile(sigma_pce_v, pcts)

print(f"|P_PI-P_MBE| percentiles (5/25/50/75/95): {[f'{v:.1f}' for v in diff_pcts]} psi")
print(f"sigma_PCE percentiles   (5/25/50/75/95): {[f'{v:.2f}' for v in sigma_pcts]} psi")
print(f"Median divergence / median sigma_PCE = {diff_pcts[2]/sigma_pcts[2]:.1f}x "
      f"-- inter-model divergence dominates intra-method uncertainty")

J_effs = df_drift["J_eff"].values
J_eff_pcts = np.percentile(J_effs, pcts)
print(f"Drift events: {len(df_drift)}  |  J_eff range=[{J_effs.min():.1f}, {J_effs.max():.1f}] STB/d/psi")
print(f"J_eff percentiles (5/25/50/75/95): {[f'{v:.1f}' for v in J_eff_pcts]}")

"""Real vs digital-twin input distributions"""

twin_pavg = pd.read_csv(os.path.join(out_dir, "digital_twin_true_Pavg.csv"))
twin_dt   = pd.read_csv(os.path.join(out_dir, "Pavg_estimates.csv"))
twin_dt["day"] = (twin_dt["time_min"] // 1440).astype(int)
twin_daily = twin_dt.groupby("day").agg(
    P_mean=("P_1min_psi", "mean"), q_mean=("q_1min_STBm", "mean"),
    state_mode=("state", lambda x: int(x.mode().iloc[0])),
).reset_index()
twin_daily["q_daily"] = twin_daily["q_mean"] * 1440.0

twin_pi_mask = twin_daily["state_mode"].values == 0
twin_q_pi    = twin_daily["q_daily"].values[twin_pi_mask]
twin_P_pi    = twin_daily["P_mean"].values[twin_pi_mask]
twin_true_pi = twin_pavg["Pavg_true"].values[:len(twin_daily)][twin_pi_mask]
twin_dd_pi   = twin_true_pi - twin_P_pi

real_q_pi  = q_arr[st_arr == 0]
real_P_pi  = P_arr[st_arr == 0]
real_dd_pi = Pavg_ref[st_arr == 0] - real_P_pi

print(f"Twin PI days:  q_mean={twin_q_pi.mean():.0f} STB/day  dd_mean={twin_dd_pi.mean():.1f} psi  "
      f"sig_P/dd={sig_P/twin_dd_pi.mean()*100:.2f}%")
print(f"Real PI days:  q_mean={real_q_pi.mean():.0f} STB/day  dd_mean={real_dd_pi.mean():.1f} psi  "
      f"sig_P/|dd|={sig_P/abs(real_dd_pi.mean())*100:.2f}%")

pd.DataFrame({
    "dataset": ["twin_PI"] * len(twin_q_pi) + ["real_PI"] * len(real_q_pi),
    "q_STBday": list(twin_q_pi) + list(real_q_pi),
    "BHP_psi": list(twin_P_pi) + list(real_P_pi),
    "drawdown_psi": list(twin_dd_pi) + list(real_dd_pi),
}).to_csv(os.path.join(out_dir, "real_vs_twin_inputs.csv"), index=False)
print("Saved: outputs/real_vs_twin_inputs.csv")

"""Figure 13: real-data UQ time series"""

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 300})

# Same colour_s mapping as Figures 7/8 (eps_P=red, eps_q=orange, J=blue, ct=green)
RED, ORANGE, BLUE, GREEN = "#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"
GREY, AMBER = "#9E9E9E", "#FF6F00"
LBLUE, LRED = "#BBDEFB", "#FFCDD2"

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Real Volve F-14 H: PCE Stochastic Pavg Estimates (3,056 Days)",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.fill_between(dates, Pavg_p025, Pavg_p975, color=LRED, alpha=0.5, label="Total 95% CI")
ax.fill_between(dates, Pavg_p025_epi, Pavg_p975_epi, color=LBLUE, alpha=0.6, label="Epistemic 95% CI")
ax.plot(dates, Pavg_ref,  color=BLUE, lw=1.2, label="Pavg MBE reference")
ax.plot(dates, Pavg_mean, color=RED,  lw=0.8, alpha=0.8, label="PCE mean")
for j, ev in enumerate(drift_events[:1]):
    ax.axvline(x=pd.Timestamp(ev["date"]), color=AMBER, lw=0.7, ls="--",
               label=f"J drift events (n={len(drift_events)})")
for ev in drift_events[1:]:
    ax.axvline(x=pd.Timestamp(ev["date"]), color=AMBER, lw=0.7, ls="--", alpha=0.4)
ax.set_ylabel("Pavg (psi)", fontsize=10)
ax.legend(fontsize=8, loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)
ax.set_title("(A) Reservoir Pressure with Stochastic Uncertainty Bands", fontsize=10)

ax2, ax2b = axes[1], axes[1].twinx()
ax2.plot(dates, Pavg_std, color=RED, lw=0.8, label="PCE std dev")
ax2.set_ylabel("Pavg std dev (psi)", fontsize=10, color=RED)
ax2.tick_params(axis="y", labelcolor=RED)
ax2b.plot(dates, J_eff_arr, color=AMBER, lw=1.0, alpha=0.9, label="J_eff rolling")
ax2b.axhline(y=J_base, color="k", lw=1.0, ls=":", label=f"J_base={J_base:.1f}")
ax2b.set_ylabel("J_eff (STB/d/psi)", fontsize=10, color=AMBER)
ax2b.tick_params(axis="y", labelcolor=AMBER)
lines_a, labs_a = ax2.get_legend_handles_labels()
lines_b, labs_b = ax2b.get_legend_handles_labels()
ax2.legend(lines_a + lines_b, labs_a + labs_b, fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3)
ax2.set_title("(B) PCE Standard Deviation and Rolling J Recalibration", fontsize=10)

ax = axes[2]
ci_tot_w = Pavg_p975 - Pavg_p025
ci_epi_w = Pavg_p975_epi - Pavg_p025_epi
ci_ale_w = np.maximum(ci_tot_w - ci_epi_w, 0.0)
ax.stackplot(dates, ci_ale_w, ci_epi_w, colors=[RED, BLUE], alpha=0.7,
             labels=["Aleatory CI (sensor noise)", "Epistemic CI (J, ct)"])
ax.set_ylabel("95% CI width (psi)", fontsize=10)
ax.set_xlabel("Date", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_title("(C) Aleatory vs Epistemic Uncertainty Decomposition", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig13_real_uq_timeseries.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig13_real_uq_timeseries.png")

"""Figure 14: time-varying Sobol' attribution map"""

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Real Volve F-14 H: Time-Varying Sobol Sensitivity Attribution",
             fontsize=13, fontweight="bold")

def smooth21(col):
    return pd.Series(col).rolling(21, min_periods=1, center=True).mean().values

s1_eP, s1_eq = smooth21(Sobol_S1[:, 0]), smooth21(Sobol_S1[:, 1])
s1_J,  s1_ct = smooth21(Sobol_S1[:, 2]), smooth21(Sobol_S1[:, 3])

ax = axes[0]
stk = np.stack([np.nan_to_num(s1_eP), np.nan_to_num(s1_eq),
                 np.nan_to_num(s1_J),  np.nan_to_num(s1_ct)], axis=0)
ax.stackplot(dates, stk[0], stk[1], stk[2], stk[3],
             colors=[RED, ORANGE, BLUE, GREEN], alpha=0.85,
             labels=["ep_P (BHP sensor)", "ep_q (flow meter)",
                     "J (PI uncertainty)", "ct (compressibility)"])
ax.set_ylabel("First-order Sobol S1", fontsize=10)
ax.set_ylim(0, 1)
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.grid(True, alpha=0.3)
mbe_mask = pce_method == "MBE"
y0_a, y1_a = ax.get_ylim()
ax.fill_between(dates, y0_a, y1_a, where=mbe_mask, color="lightgrey", alpha=0.4, label="MBE days")
ax.set_title("(A) First-Order Sobol Indices (21-day rolling average)", fontsize=10)

ax = axes[1]
dom_colors_map = {"ep_P (BHP noise)": RED, "ep_q (flow noise)": ORANGE,
                   "J (PI uncert.)": BLUE, "ct (compress.)": GREEN, "unknown": GREY}
for j, d in enumerate(dates):
    ax.bar(d, 1, width=1.0, color=dom_colors_map.get(dominant[j], GREY), alpha=0.7)
ax.set_yticks([])
ax.set_ylabel("Dominant source", fontsize=10)
ax.set_xlabel("Date", fontsize=10)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=RED,    label="ep_P dominant"),
    Patch(color=ORANGE, label="ep_q dominant"),
    Patch(color=BLUE,   label="J dominant"),
    Patch(color=GREEN,  label="ct dominant"),
], fontsize=9, loc="upper right", ncol=2)
ax.set_title("(B) Dominant Uncertainty Source per Day", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig14_real_sobol_map.png"), bbox_inches="tight")
plt.close(fig)
print("Saved: outputs/fig14_real_sobol_map.png")
