import os
import numpy as np
import pandas as pd

folder  = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(folder, "outputs")
os.makedirs(out_dir, exist_ok=True)
np.random.seed(42)

BAR2PSI = 14.5038
SM3_STB = 6.2898

params   = pd.read_csv(os.path.join(out_dir, "F14_parameters.csv"), index_col="Parameter")["Value"]
Pi       = params["Pi"]        # psi
J0       = params["J0"]        # STB/day/psi
Np_total = params["Np_total"]  # STB, whole 3,056-day production history

ct         = 15e-6              
Pavg_final = 200.0 * BAR2PSI    
N_ooip     = Np_total / (ct * (Pi - Pavg_final))

# Real Volve F-14 daily oil volumes, first 365 producing days
volve = pd.read_excel(os.path.join(folder, "data", "Volve production data.xlsx"),
                       sheet_name="Daily Production Data")
well  = volve[volve["NPD_WELL_BORE_NAME"] == "15/9-F-14"].sort_values("DATEPRD").reset_index(drop=True)
well_prod = well[well["BORE_OIL_VOL"] > 0].reset_index(drop=True)

N_DAYS  = min(365, len(well_prod))
q_daily = well_prod["BORE_OIL_VOL"].values[:N_DAYS].astype(float) * SM3_STB   # STB/day

SECS_PER_DAY = 86400
MINS_PER_DAY = 1440
N_SECONDS    = N_DAYS * SECS_PER_DAY
N_MINUTES    = N_DAYS * MINS_PER_DAY
sigma_P      = 2.0    # psi, PDHMS noise
sigma_q      = 0.02   # fractional, MPFM noise

# MPFM transport lag: random draw from U(15,45) minutes, reproducible via seed
rng      = np.random.default_rng(seed=7)
LAG_MINS = int(rng.integers(15, 46))

# Scheduled shut-ins
shutin_starts  = [30, 75, 120, 165, 210, 255, 300, 345]
shutin_lengths = [4,   3,   5,   4,   3,   5,   4,   3]
for start, length in zip(shutin_starts, shutin_lengths):
    q_daily[start:start + length] = 0.0

Np_daily   = np.cumsum(q_daily)
Pavg_daily = Pi - Np_daily / (N_ooip * ct)

# 1-second PDHMS pressure stream: Pwf = Pavg - q/J0, capped at Pwf >= 14.7 psi (1 atm)
Pwf_seconds = np.zeros(N_SECONDS)
n_capped = 0
for day in range(N_DAYS):
    s, e = day * SECS_PER_DAY, (day + 1) * SECS_PER_DAY
    Pavg_now, q_now = Pavg_daily[day], q_daily[day]
    if q_now > 0:
        q_phys = min(q_now, J0 * max(0.0, Pavg_now - 14.7))
        if q_phys < q_now:
            n_capped += 1
        Pwf_base = Pavg_now - q_phys / J0
        noise    = np.random.normal(0, sigma_P, SECS_PER_DAY)
        Pwf_seconds[s:e] = np.maximum(Pwf_base + noise, 14.7)
    else:
        prev_Pwf = Pwf_seconds[s - 1] if s > 0 else Pavg_now * 0.85
        t_hrs    = np.arange(SECS_PER_DAY) / 3600
        buildup  = Pavg_now - (Pavg_now - prev_Pwf) * np.exp(-0.5 * t_hrs)
        noise    = np.random.normal(0, sigma_P * 0.3, SECS_PER_DAY)
        Pwf_seconds[s:e] = buildup + noise

# Injected PDHMS artifacts: 3 spikes, 1 freeze, 1 drift ramp
for t in [50000, 150000, 250000]:
    Pwf_seconds[t:t + 60] += 250.0
Pwf_seconds[100000:107200] = Pwf_seconds[100000]        # 2 hr freeze
Pwf_seconds[201600:216000] += np.linspace(0, 60, 14400)  # 4 hr drift ramp

# 1-minute MPFM flow rate stream
q_minutes = np.zeros(N_MINUTES)
for day in range(N_DAYS):
    s, e = day * MINS_PER_DAY, (day + 1) * MINS_PER_DAY
    q_now = q_daily[day]
    if q_now > 0:
        q_per_min = q_now / MINS_PER_DAY
        noise     = np.random.normal(0, sigma_q * q_per_min, MINS_PER_DAY)
        t_min     = np.arange(MINS_PER_DAY)
        slug      = 0.03 * q_per_min * np.sin(2 * np.pi * t_min / 60)
        q_minutes[s:e] = np.maximum(0.0, q_per_min + noise + slug)

# Injected MPFM artifacts: systematic bias, data gap
q_minutes[2000:2030] *= 1.20
q_minutes[5000:5030]  = np.nan

# Transport lag: q_lagged[t] = q_minutes[t - LAG_MINS]
q_lagged            = np.empty_like(q_minutes)
q_lagged[:LAG_MINS]  = q_minutes[0]
q_lagged[LAG_MINS:]  = q_minutes[:-LAG_MINS]

pd.DataFrame({"time_seconds": np.arange(N_SECONDS), "Pwf_psi": Pwf_seconds}
             ).to_csv(os.path.join(out_dir, "digital_twin_PDHMS.csv"), index=False)
pd.DataFrame({"time_minutes": np.arange(N_MINUTES), "q_STB_per_min": q_lagged}
             ).to_csv(os.path.join(out_dir, "digital_twin_MPFM.csv"), index=False)
pd.DataFrame({"day": np.arange(N_DAYS), "Pavg_true": Pavg_daily, "q_daily": q_daily}
             ).to_csv(os.path.join(out_dir, "digital_twin_true_Pavg.csv"), index=False)
pd.DataFrame({"Parameter": ["LAG_MINS", "sigma_P", "sigma_q", "ct", "N_ooip"],
              "Value":     [LAG_MINS, sigma_P, sigma_q, ct, N_ooip]}
             ).to_csv(os.path.join(out_dir, "digital_twin_params.csv"), index=False)

print(f"N_ooip     = {N_ooip:.0f} STB")
print(f"LAG_MINS   = {LAG_MINS} min")
print(f"n_capped   = {n_capped} / {N_DAYS} days")
print(f"Pavg range = [{Pavg_daily.min():.1f}, {Pavg_daily.max():.1f}] psi")
print("Saved: outputs/digital_twin_{PDHMS,MPFM,true_Pavg,params}.csv")
