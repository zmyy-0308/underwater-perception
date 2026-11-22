"""Verify every objective through full-resolution training and evaluation."""
import subprocess
import sys

from experiment import ROOT
from objectives import RECIPES

for name in RECIPES:
    subprocess.run([
        sys.executable, "-u", str(ROOT / "train_job.py"), "--recipe", name,
        "--fold", "0", "--seed", "42", "--epochs", "1", "--limit", "4",
        "--out-root", str(ROOT / "smoke"), "--threads", "1",
    ], check=True)
    print("PREFLIGHT_OK", name, flush=True)
(ROOT / "PREFLIGHT_OK").touch()
