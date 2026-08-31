import os
import pandas as pd

folder  = os.path.dirname(os.path.abspath(__file__))
data_fp = os.path.join(folder, "data", "Volve production data.xlsx")
out_dir = os.path.join(folder, "outputs")
os.makedirs(out_dir, exist_ok=True)

BAR2PSI = 14.5038   # 1 bar = 14.5038 psi  (Volve pressure is in bar)
SM3_STB = 6.2898    # 1 Sm3 = 6.2898 STB   (Volve volumes are in Sm3)

df   = pd.read_excel(data_fp, sheet_name="Daily Production Data")
well = df[df["NPD_WELL_BORE_NAME"] == "15/9-F-14"].sort_values("DATEPRD").reset_index(drop=True)

bhp_valid   = well["AVG_DOWNHOLE_PRESSURE"] > 0
oil_valid   = well["BORE_OIL_VOL"] > 0
prod_mask   = bhp_valid & oil_valid
shutin_mask = bhp_valid & ~oil_valid
prod        = well[prod_mask]

Pi       = well["AVG_DOWNHOLE_PRESSURE"].max() * BAR2PSI
Pwf_mean = prod["AVG_DOWNHOLE_PRESSURE"].mean() * BAR2PSI
drawdown = Pi - Pwf_mean
q_mean   = prod["BORE_OIL_VOL"].mean() * SM3_STB
J0       = q_mean / drawdown
Np_total = well["BORE_OIL_VOL"].fillna(0).sum() * SM3_STB

print(f"Pi        = {Pi:.4f} psi")
print(f"Pwf_mean  = {Pwf_mean:.4f} psi")
print(f"drawdown  = {drawdown:.4f} psi")
print(f"J0        = {J0:.4f} STB/day/psi")
print(f"Np_total  = {Np_total:.0f} STB")
print(f"n_prod    = {int(prod_mask.sum())}")
print(f"n_shutin  = {int(shutin_mask.sum())}")

params = pd.DataFrame({
    "Parameter": ["Pi", "mean_bhp", "drawdown", "J0", "Np_total"],
    "Value":     [Pi, Pwf_mean, drawdown, J0, Np_total],
})
params.to_csv(os.path.join(out_dir, "F14_parameters.csv"), index=False)
print("Saved: outputs/F14_parameters.csv")
