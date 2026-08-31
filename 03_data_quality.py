import os
import numpy as np
import pandas as pd
from scipy.signal import correlate

folder  = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(folder, "outputs")

QC = {
    "P_min": 14.7, "P_max": 8000.0,             # psi -- 1 atm to well above Pi
    "spike_dP": 200.0,                           # psi/s, 1-second PDHMS spike
    "roc_dP_permin": 100.0,                      # psi/min, on 1-min averages
    "freeze_win": 3600, "freeze_std": 0.05,      # samples (60 min @ 1 Hz), psi
    "q_min": 0.0, "q_max": 50.0,                 # STB/min
    "drift_win": 14400, "drift_slope": 0.002,    # samples (4 hr), psi/s
}

pdhms = pd.read_csv(os.path.join(out_dir, "digital_twin_PDHMS.csv"))
mpfm  = pd.read_csv(os.path.join(out_dir, "digital_twin_MPFM.csv"))
pavg  = pd.read_csv(os.path.join(out_dir, "digital_twin_true_Pavg.csv"))
dt_params = pd.read_csv(os.path.join(out_dir, "digital_twin_params.csv"), index_col="Parameter")["Value"]
TRUE_LAG  = int(dt_params["LAG_MINS"])   # reference only, never used to correct

P_raw = pdhms["Pwf_psi"].values.copy()
q_raw = mpfm["q_STB_per_min"].values.copy()
Pavg_true_daily = pavg["Pavg_true"].values

n = len(P_raw)
flag_P = np.zeros(n, dtype=int)

# Check 1: range
range_mask = (P_raw < QC["P_min"]) | (P_raw > QC["P_max"])
flag_P[range_mask] = 1

# Check 2: spike
dP_per_sec = np.abs(np.diff(P_raw, prepend=P_raw[0]))
spike_mask = dP_per_sec > QC["spike_dP"]
flag_P[spike_mask] = 1

# Check 3: freeze -- rolling std < freeze_std over freeze_win seconds.
# Vectorized via cumsum run-length: flag a frozen minute if it belongs to a
# run of >= roll_win_min consecutive frozen minutes, extended backward to the
# start of that run.
SECS_PER_MIN = 60
n_full_min   = n // SECS_PER_MIN
P_min_std    = P_raw[:n_full_min * SECS_PER_MIN].reshape(n_full_min, SECS_PER_MIN).std(axis=1)
roll_win_min = QC["freeze_win"] // SECS_PER_MIN
min_frozen   = P_min_std < QC["freeze_std"]
cum          = np.cumsum(min_frozen.astype(np.int32))
padded_cum   = np.concatenate([np.zeros(roll_win_min, dtype=np.int32), cum])
run_counts   = cum - padded_cum[:n_full_min]
freeze_mask_1min = np.zeros(n_full_min, dtype=bool)
for idx in np.where(run_counts >= roll_win_min)[0]:
    freeze_mask_1min[max(0, idx - roll_win_min + 1):idx + 1] = True
freeze_mask_1min &= min_frozen
freeze_mask = np.zeros(n, dtype=bool)
freeze_mask[:n_full_min * SECS_PER_MIN] = np.repeat(freeze_mask_1min, SECS_PER_MIN)
flag_P[freeze_mask] = 1

# Check 4: rate of change, on 1-minute averages (skip samples already flagged as spikes)
P_1min_avg    = P_raw[:n_full_min * SECS_PER_MIN].reshape(n_full_min, SECS_PER_MIN).mean(axis=1)
roc_1min_mask = np.abs(np.diff(P_1min_avg, prepend=P_1min_avg[0])) > QC["roc_dP_permin"]
roc_sec_mask  = np.zeros(n, dtype=bool)
roc_sec_mask[:n_full_min * SECS_PER_MIN] = np.repeat(roc_1min_mask, SECS_PER_MIN)
roc_sec_mask &= ~spike_mask
flag_P[roc_sec_mask] = 1

P_clean = P_raw.astype(float).copy()
P_clean[flag_P == 1] = np.nan
P_clean = pd.Series(P_clean).interpolate(method="linear", limit_direction="both").values

# Drift: vectorized OLS slope per non-overlapping drift_win window,
# slope = (x_c . y_c) / (x_c . x_c)
drift_win = QC["drift_win"]
n_windows = n // drift_win
x         = np.arange(drift_win, dtype=np.float32)
x_c       = (x - x.mean()).astype(np.float32)
x_sq      = float(np.dot(x_c, x_c))
P_wins    = P_raw[:n_windows * drift_win].reshape(n_windows, drift_win).astype(np.float32)
P_wins_c  = P_wins - P_wins.mean(axis=1, keepdims=True)
drift_slopes = P_wins_c.dot(x_c) / x_sq
drift_flagged_wins = np.abs(drift_slopes) > QC["drift_slope"]

# MPFM QC
nq = len(q_raw)
flag_q = np.zeros(nq, dtype=int)
nan_mask = np.isnan(q_raw)
flag_q[nan_mask] = 1
flag_q[(~nan_mask) & ((q_raw < QC["q_min"]) | (q_raw > QC["q_max"]))] = 1

q_tmp   = q_raw.copy()
q_tmp[nan_mask] = np.nanmedian(q_raw)
dq      = np.abs(np.diff(q_tmp, prepend=q_tmp[0]))
spike_q = (~nan_mask) & (dq > 0.5 * np.nanmedian(q_raw))
flag_q[spike_q] = 1

q_clean = q_raw.astype(float).copy()
q_clean[flag_q == 1] = np.nan
q_clean = pd.Series(q_clean).interpolate(method="linear", limit=30, limit_direction="forward").values
q_clean = pd.Series(q_clean).bfill().values

# Blind transport-lag detection: cross-correlate -P (anti-correlated with q)
# against q on the first 30 days, search +-120 min. TRUE_LAG is never used here.
P_1min = P_clean[:n_full_min * SECS_PER_MIN].reshape(n_full_min, SECS_PER_MIN).mean(axis=1)
q_1min = q_clean[:n_full_min]
MAX_SEARCH_MINS = min(n_full_min, 30 * 1440)
P_seg = P_1min[:MAX_SEARCH_MINS]
q_seg = q_1min[:MAX_SEARCH_MINS]
P_n = -(P_seg - P_seg.mean()) / (P_seg.std() + 1e-9)
q_n =  (q_seg - q_seg.mean()) / (q_seg.std() + 1e-9)
xcorr = correlate(q_n, P_n, mode="full")
mid   = len(xcorr) // 2
MAX_LAG_SEARCH = 120
win_xcorr = xcorr[mid - MAX_LAG_SEARCH:mid + MAX_LAG_SEARCH + 1]
win_lags  = np.arange(-MAX_LAG_SEARCH, MAX_LAG_SEARCH + 1)
DETECTED_LAG = int(win_lags[np.argmax(win_xcorr)])

# q_corrected[t] = q_clean[t + LAG]: advance q to undo the transport delay
LAG_APPLY = max(0, DETECTED_LAG)
q_lag_corrected = np.empty_like(q_clean)
if LAG_APPLY > 0:
    q_lag_corrected[:-LAG_APPLY] = q_clean[LAG_APPLY:]
    q_lag_corrected[-LAG_APPLY:] = q_clean[-1]
else:
    q_lag_corrected[:] = q_clean

# Cross-stream validation: Pwf must stay below Pavg (1 psi tolerance)
n_days_full = n // 86400
Pwf_daily = P_clean[:n_days_full * 86400].reshape(n_days_full, 86400).mean(axis=1)
Pavg_ref  = Pavg_true_daily[:n_days_full]
violations = Pwf_daily > (Pavg_ref + 1.0)

# Save: PDHMS downsampled to 1-minute resolution (mean pressure, max flag per minute)
P_clean_1min = P_clean[:n_full_min * SECS_PER_MIN].reshape(n_full_min, SECS_PER_MIN).mean(axis=1)
flag_1min    = flag_P[:n_full_min * SECS_PER_MIN].reshape(n_full_min, SECS_PER_MIN).max(axis=1)
pd.DataFrame({
    "time_minutes": np.arange(n_full_min),
    "Pwf_clean": P_clean_1min,
    "QC_flag": flag_1min,
}).to_csv(os.path.join(out_dir, "PDHMS_clean.csv"), index=False)

mpfm_out = mpfm.copy()
mpfm_out["q_clean"] = q_clean
mpfm_out["q_lag_corrected"] = q_lag_corrected
mpfm_out["QC_flag"] = flag_q
mpfm_out.to_csv(os.path.join(out_dir, "MPFM_clean.csv"), index=False)

pd.DataFrame({
    "Stream": ["PDHMS (1-min agg)", "MPFM (1-min)"],
    "Total_Samples": [n_full_min, nq],
    "Flagged": [int(flag_1min.sum()), int(flag_q.sum())],
    "Pct_Flagged": [f"{100 * flag_P.sum() / n:.4f}%", f"{100 * flag_q.sum() / nq:.4f}%"],
    "Drift_Windows": [int(drift_flagged_wins.sum()), "N/A"],
    "Lag_Detected_min": [DETECTED_LAG, "N/A"],
    "Lag_Applied_min": [LAG_APPLY, "N/A"],
    "Crossstream_Viol": [int(violations.sum()), "N/A"],
}).to_csv(os.path.join(out_dir, "QC_flag_summary.csv"), index=False)

print(f"Range/Spike/Freeze/RoC flagged = {range_mask.sum()}/{spike_mask.sum()}/{freeze_mask.sum()}/{roc_sec_mask.sum()}")
print(f"PDHMS flagged  = {flag_P.sum()} / {n} ({100 * flag_P.sum() / n:.4f}%)")
print(f"MPFM flagged   = {flag_q.sum()} / {nq} ({100 * flag_q.sum() / nq:.4f}%)")
print(f"Drift windows  = {int(drift_flagged_wins.sum())} / {n_windows}")
print(f"Lag detected   = {DETECTED_LAG} min (true = {TRUE_LAG} min)")
print(f"Cross-stream violations = {int(violations.sum())}")
print("Saved: outputs/{PDHMS_clean,MPFM_clean,QC_flag_summary}.csv")
