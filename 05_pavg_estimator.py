import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # This is because some daily windows have no State-0 minute

folder  = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(folder, "outputs")

BAR2PSI = 14.5038
SM3_STB = 6.2898

params   = pd.read_csv(os.path.join(out_dir, "F14_parameters.csv"), index_col="Parameter")["Value"]
Pi       = float(params["Pi"])
J0       = float(params["J0"])
ct       = 15.0e-6

dt_params = pd.read_csv(os.path.join(out_dir, "digital_twin_params.csv"), index_col="Parameter")["Value"]
N_ooip    = float(dt_params["N_ooip"])

J_DRIFT_THR  = 0.15     # flag if J deviates >15% from J0
RECALIB_WIN  = 48 * 60  # 48 hours, in minutes
DRAWDOWN_MIN = 100.0    # psi, minimum drawdown admitted to J recalibration (matches 07/08)

df_states = pd.read_csv(os.path.join(out_dir, "operational_states.csv"))
df_pavg   = pd.read_csv(os.path.join(out_dir, "digital_twin_true_Pavg.csv"))

t_min  = df_states["time_min"].values
P_1min = df_states["P_1min_psi"].values
q_1min = df_states["q_1min_STBm"].values
state  = df_states["state"].values
t_day  = t_min / 1440.0

Pavg_true    = df_pavg["Pavg_true"].values
q_daily_true = df_pavg["q_daily"].values   # STB/day, same rates the twin used for Pwf

# Method 1: PI with rolling 48-hr J recalibration (paper Eq. 1).
# MBE Pavg is used as the recalibration reference.
N = len(t_min)
Pavg_PI      = np.full(N, np.nan)
J_effective  = np.full(N, J0)
J_drift_flag = np.zeros(N, dtype=bool)
stable_mask  = state == 0

Np_lookup    = np.cumsum(q_1min)
Pavg_MBE_ref = np.clip(Pi - Np_lookup / (N_ooip * ct), 0, Pi)

# Daily production-report rate
day_idx  = np.minimum((t_min / 1440.0).astype(int), len(q_daily_true) - 1)
q_for_pi = q_daily_true[day_idx]

in_stable, J_current = False, J0
for i in range(N):
    if stable_mask[i] and not in_stable:
        in_stable = True
        look_back = max(0, i - RECALIB_WIN)
        win_mask  = stable_mask[look_back:i]
        if win_mask.sum() >= 60:
            q_win     = q_for_pi[look_back:i][win_mask]
            Pwf_win   = P_1min[look_back:i][win_mask]
            P_ref_win = Pavg_MBE_ref[look_back:i][win_mask]
            draw      = P_ref_win - Pwf_win
            pos       = draw > DRAWDOWN_MIN
            if pos.sum() >= 10:
                J_candidate = np.nanmedian(q_win[pos] / draw[pos])
                if J_candidate > 0:
                    J_current = J_candidate
    elif not stable_mask[i]:
        in_stable = False
    J_effective[i] = J_current
    if abs(J_current - J0) / J0 > J_DRIFT_THR:
        J_drift_flag[i] = True

producing_mask = stable_mask & (q_for_pi > 0)
Pavg_PI[producing_mask] = P_1min[producing_mask] + (q_for_pi[producing_mask] / J_effective[producing_mask])
Pavg_PI = np.clip(Pavg_PI, 0, Pi)

# Method 2: MBE cross-validation (paper Eq. 2)
Np_cumulative = np.cumsum(q_1min)
Pavg_MBE = np.clip(Pi - Np_cumulative / (N_ooip * ct), 0, Pi)

def metrics(y_pred, y_true):
    yp, yt = np.asarray(y_pred, dtype=float), np.asarray(y_true, dtype=float)
    mask   = ~(np.isnan(yp) | np.isnan(yt))
    yp, yt = yp[mask], yt[mask]
    e  = yp - yt
    r2 = 1.0 - np.sum(e ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-12)
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2))), float(r2)

n_full_days = min(len(Pavg_true), len(Pavg_MBE) // 1440, N // 1440)
Pavg_MBE_d  = Pavg_MBE[:n_full_days * 1440].reshape(n_full_days, 1440).mean(axis=1)
Pavg_PI_d   = np.array([np.nanmean(Pavg_PI[d * 1440:(d + 1) * 1440]) for d in range(n_full_days)])
nc = min(len(Pavg_true), len(Pavg_MBE_d), len(Pavg_PI_d))

MAE_MBE, RMSE_MBE, R2_MBE = metrics(Pavg_MBE_d[:nc], Pavg_true[:nc])
MAE_PI,  RMSE_PI,  R2_PI  = metrics(Pavg_PI_d[:nc],  Pavg_true[:nc])

# Real Volve F-14 daily Pavg
volve = pd.read_excel(os.path.join(folder, "data", "Volve production data.xlsx"),
                       sheet_name="Daily Production Data")
f14   = volve[volve["NPD_WELL_BORE_NAME"] == "15/9-F-14"].sort_values("DATEPRD").reset_index(drop=True)

bhp_bar = f14["AVG_DOWNHOLE_PRESSURE"].fillna(0).values.astype(float)
oil_sm3 = f14["BORE_OIL_VOL"].fillna(0).values.astype(float)
bhp_psi = bhp_bar * BAR2PSI
oil_stb = oil_sm3 * SM3_STB

bhp_valid = bhp_bar > 0
producing = bhp_valid & (oil_stb > 0)
shutin    = bhp_valid & ~(oil_stb > 0)

# QC pass on raw daily BHP (producing days only): flags isolated dropouts via a
# 7-day rolling median and interpolates over them.
BHP_QC_WINDOW = 7
BHP_QC_THRESHOLD = 800.0  # psi

prod_idx = np.where(producing)[0]
bhp_prod = pd.Series(bhp_psi[prod_idx])
roll_med = bhp_prod.rolling(window=BHP_QC_WINDOW, center=True, min_periods=4).median()
is_outlier_prod = ((bhp_prod - roll_med).abs() > BHP_QC_THRESHOLD).values

n_bhp_outliers = int(is_outlier_prod.sum())
if n_bhp_outliers > 0:
    bhp_prod_clean = bhp_prod.copy()
    bhp_prod_clean[is_outlier_prod] = np.nan
    bhp_prod_clean = bhp_prod_clean.interpolate(method="linear", limit_direction="both")
    bhp_psi[prod_idx] = bhp_prod_clean.values
print(f"BHP QC: flagged and interpolated {n_bhp_outliers} producing-day outlier(s) "
      f"(|dev| > {BHP_QC_THRESHOLD:.0f} psi from {BHP_QC_WINDOW}-day rolling median)")

Np_cum_volve   = np.cumsum(oil_stb)
Pavg_MBE_volve = np.clip(Pi - Np_cum_volve / (N_ooip * ct), 0, Pi)

pd.DataFrame({
    "date": f14["DATEPRD"].values,
    "q_oil_STB_day": oil_stb,
    "BHP_psi": np.where(bhp_valid, bhp_psi, np.nan),
    "Pavg_MBE_psi": Pavg_MBE_volve,
    "producing_day": producing.astype(int),
    "shutin_day": shutin.astype(int),
}).to_csv(os.path.join(out_dir, "Volve_F14_Pavg_daily.csv"), index=False)

# Save digital-twin outputs
pd.DataFrame({
    "time_min": t_min, "time_day": t_day, "P_1min_psi": P_1min, "q_1min_STBm": q_1min,
    "state": state, "Pavg_PI": Pavg_PI, "Pavg_MBE": Pavg_MBE,
    "J_effective": J_effective, "J_drift_flag": J_drift_flag.astype(int),
    "Np_cum_STB": Np_cumulative,
}).to_csv(os.path.join(out_dir, "Pavg_estimates.csv"), index=False)

pd.DataFrame([
    {"Method": "PI Method (clipped)", "MAE_psi": round(MAE_PI, 2),  "RMSE_psi": round(RMSE_PI, 2),  "R2": round(R2_PI, 3)},
    {"Method": "MBE",                 "MAE_psi": round(MAE_MBE, 2), "RMSE_psi": round(RMSE_MBE, 2), "R2": round(R2_MBE, 3)},
]).to_csv(os.path.join(out_dir, "error_summary.csv"), index=False)

print(f"PI  error = MAE={MAE_PI:.2f} psi, RMSE={RMSE_PI:.2f} psi, R2={R2_PI:.3f}")
print(f"MBE error = MAE={MAE_MBE:.2f} psi, RMSE={RMSE_MBE:.2f} psi, R2={R2_MBE:.3f}")
print(f"Drift-flagged samples = {int(J_drift_flag.sum())}")
print(f"Real F-14: producing={int(producing.sum())}, shut-in={int(shutin.sum())}, "
      f"Pavg range=[{Pavg_MBE_volve.min():.1f}, {Pavg_MBE_volve.max():.1f}] psi")
print("Saved: outputs/{Pavg_estimates,error_summary,Volve_F14_Pavg_daily}.csv")
