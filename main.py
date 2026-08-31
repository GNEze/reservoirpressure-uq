import subprocess, sys, os

folder  = os.path.dirname(os.path.abspath(__file__))
scripts = [
    "01_volve_data_exploration.py",
    "02_digital_twin.py",
    "03_data_quality.py",
    "04_state_classifier.py",
    "05_pavg_estimator.py",
    "06_pce_uq_digital_twin.py",
    "07_pce_uq_real_data.py",
    "08_bma_layer.py",
]

for s in scripts:
    print(f"Running {s} ...")
    result = subprocess.run([sys.executable, os.path.join(folder, s)])
    if result.returncode != 0:
        print(f"{s} failed — stopping.")
        sys.exit(1)

print("Pipeline complete.")
