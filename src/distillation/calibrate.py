"""Measure total completed epochs at several parallelism settings."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from experiment import ROOT, write_json

RECIPES = ("pixel_kl", "pixel_kl_feature_cwd", "tailor_joint", "feature_cwd_multiscale")
JOBS = [(name, fold) for fold in (0, 1) for name in RECIPES]


def measure(slots, stop):
    def one(job):
        name, fold = job
        path = ROOT / "runs" / name / f"f{fold}-s42"
        file = ROOT / "calibration" / f"{name}-f{fold}-{slots}.log"
        file.parent.mkdir(exist_ok=True)
        with file.open("w") as stream:
            subprocess.run([
                sys.executable, "-u", str(ROOT / "train_job.py"), "--recipe", name,
                "--fold", str(fold), "--seed", "42", "--threads", "1",
                "--stop-after", str(stop),
            ], stdout=stream, stderr=subprocess.STDOUT, check=True)
        history = json.loads((path / "history.json").read_bytes())
        return history[-1]["seconds"]
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=slots) as pool:
        timings = list(pool.map(one, JOBS))
    # Runtime includes model load and checkpoint IO. Retain both throughput estimates.
    elapsed = perf_counter() - start
    return {"slots": slots, "wall_seconds": elapsed,
            "steady_epochs_per_minute": 60 * len(timings) / (sum(timings) / slots),
            "last_epoch_seconds": timings}


if __name__ == "__main__":
    measurements = [measure(4, 2), measure(8, 4)]
    chosen = max(measurements, key=lambda row: row["steady_epochs_per_minute"])["slots"]
    write_json(ROOT / "calibration.json", {"measurements": measurements, "chosen_slots": chosen})
    (ROOT / "chosen-slots").write_text(str(chosen))
    print("CALIBRATION", measurements, "CHOSEN", chosen, flush=True)
