import os
import numpy as np
import pandas as pd

folder  = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(folder, "outputs")

CFG = {
    "window_min": 30,             # minutes, sliding window
    "q_prod_thr": 100 / 1440,     # STB/min, minimum q to be "producing"
    "q_shutin_thr": 50 / 1440,    # STB/min, below this = shut-in
    "dP_stable_max": 10 / 60,     # psi/min, max |dP/dt| for stable state
    "dP_transient_min": 50 / 60,  # psi/min, min |dP/dt| for transient state
    "buildup_dP_min": 0.01,       # psi/min, min dP/dt to confirm active buildup
    "dQ_thr": 100 / 1440,         # STB/min/min, |dQ/dt| for transient onset
    "sustained_min": 120,         # minutes, min duration to confirm stable state
}
STATE_NAMES = {0: "Stable Production", 1: "Transient", 2: "Pressure Buildup", 3: "Injection"}

pdhms = pd.read_csv(os.path.join(out_dir, "PDHMS_clean.csv"))
mpfm  = pd.read_csv(os.path.join(out_dir, "MPFM_clean.csv"))

t_min  = pdhms["time_minutes"].values.astype(float)
P_1min = pdhms["Pwf_clean"].values
q_1min = mpfm["q_lag_corrected"].values

N = min(len(t_min), len(q_1min))
t_min, P_1min, q_1min = t_min[:N], P_1min[:N], q_1min[:N]

# dP/dt, dQ/dt (psi/min, STB/min/min): central finite difference over the
# window. Boundaries are held at the nearest interior value
win  = CFG["window_min"]
half = win // 2
dP_dt = np.zeros(N)
dP_dt[half:N - half] = (P_1min[win:] - P_1min[:N - win]) / win
dP_dt[:half]  = dP_dt[half]
dP_dt[-half:] = dP_dt[N - half - 1]

dQ_dt = np.zeros(N)
dQ_dt[half:N - half] = (q_1min[win:] - q_1min[:N - win]) / win
dQ_dt[:half]  = dQ_dt[half]
dQ_dt[-half:] = dQ_dt[N - half - 1]

q_roll_mean = pd.Series(q_1min).rolling(window=win, min_periods=win // 2, center=True).mean()
q_roll_mean = q_roll_mean.bfill().ffill().values

# State 2, pressure buildup: q below shut-in threshold, actively recovering.
# Flat shut-ins (q~=0, dP/dt~=0) are labelled Stable (state 0), and not Buildup.
state = np.full(N, 0, dtype=int)
buildup_mask = (q_roll_mean < CFG["q_shutin_thr"]) & (dP_dt > CFG["buildup_dP_min"])
state[buildup_mask] = 2

# State 1, transient: flowing, and pressure or rate changing rapidly
transient_mask = (q_roll_mean >= CFG["q_shutin_thr"]) & (
    (np.abs(dP_dt) >= CFG["dP_transient_min"]) | (np.abs(dQ_dt) >= CFG["dQ_thr"])
)
state[transient_mask] = 1

# State 0, stable production: q above threshold, |dP/dt| small, sustained
# >= sustained_min minutes. Candidates that don't sustain long enough are
# reclassified as Transient.
stable_cand = (
    (q_roll_mean >= CFG["q_prod_thr"]) &
    (np.abs(dP_dt) <= CFG["dP_stable_max"]) &
    (state == 0)
)
final_stable = stable_cand.copy()
in_run, run_start = False, 0
for i in range(N):
    if stable_cand[i] and not in_run:
        in_run, run_start = True, i
    elif not stable_cand[i] and in_run:
        in_run = False
        if (i - run_start) < CFG["sustained_min"]:
            final_stable[run_start:i] = False
if in_run and (N - run_start) < CFG["sustained_min"]:
    final_stable[run_start:N] = False
state[stable_cand & ~final_stable] = 1

# State 3, injection: q < 0 (not present in this dataset)
state[q_roll_mean < 0] = 3

# Buildup events: consecutive state-2 runs >= 60 min
buildup_events = []
in_b, bs = False, 0
for i in range(N):
    if state[i] == 2 and not in_b:
        in_b, bs = True, i
    elif state[i] != 2 and in_b:
        in_b = False
        if i - bs >= 60:
            buildup_events.append({"start_min": t_min[bs], "end_min": t_min[i - 1],
                                    "duration_h": (i - bs) / 60,
                                    "P_start": P_1min[bs], "P_end": P_1min[i - 1],
                                    "dP_total": P_1min[i - 1] - P_1min[bs]})
if in_b and (N - bs) >= 60:
    buildup_events.append({"start_min": t_min[bs], "end_min": t_min[N - 1],
                            "duration_h": (N - bs) / 60,
                            "P_start": P_1min[bs], "P_end": P_1min[N - 1],
                            "dP_total": P_1min[N - 1] - P_1min[bs]})
df_events = pd.DataFrame(buildup_events)

pd.DataFrame({
    "time_min": t_min, "P_1min_psi": P_1min, "q_1min_STBm": q_1min,
    "dP_dt": dP_dt, "dQ_dt": dQ_dt, "state": state,
    "state_name": [STATE_NAMES[s] for s in state],
}).to_csv(os.path.join(out_dir, "operational_states.csv"), index=False)

if len(df_events) > 0:
    df_events.to_csv(os.path.join(out_dir, "buildup_events.csv"), index=False)

for s, name in STATE_NAMES.items():
    cnt = int((state == s).sum())
    print(f"State {s} ({name:18s}) = {cnt:7,} min ({100 * cnt / N:5.1f}%) = {cnt / 1440:.1f} days")
print(f"Buildup events >= 1 hr = {len(df_events)}")
print("Saved: outputs/{operational_states,buildup_events}.csv")
