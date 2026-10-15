"""Run the full training/report pipeline, then open the Streamlit dashboard."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STEPS = [
    "01_inspect_data.py", "02_preprocess.py", "03_kmeans.py", "04_dbscan.py",
    "05_dec.py", "06_cluster_analysis.py", "07_compare_models.py",
    "08_final_visualization.py", "09_final_report.py",
]

for step in STEPS:
    script = ROOT / "src" / step
    print(f"\n=== Running {step} ===", flush=True)
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    if result.returncode:
        raise SystemExit(f"Pipeline stopped: {step} failed with exit code {result.returncode}.")

dashboard = ROOT / "src" / "11_dashboard.py"
print("\n=== Launching dashboard (Ctrl+C to stop) ===", flush=True)
try:
    result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard)], cwd=ROOT)
except KeyboardInterrupt:
    print("Dashboard stopped by user.")
    raise SystemExit(0)
if result.returncode:
    raise SystemExit(f"Pipeline outputs were generated, but dashboard failed with exit code {result.returncode}.")
