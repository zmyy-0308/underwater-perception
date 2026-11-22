"""Bounded parallel execution and live on-disk status for all 160 trials."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from experiment import ROOT, plan, write_json


class AdoptedProcess:
    def __init__(self, pid, job):
        self.pid = pid
        self.job = job
        self.start = self.info()[1]

    def info(self):
        fields = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], fields[19]

    def poll(self):
        try:
            state, start = self.info()
            if start == self.start and state != "Z":
                return None
        except (FileNotFoundError, ProcessLookupError):
            pass
        folder = ROOT / "runs" / self.job["recipe"] / f"f{self.job['fold']}-s{self.job['seed']}"
        return 0 if (folder / "COMPLETE").exists() else 130


def identity(job):
    return job["recipe"], job["fold"], job["seed"]


def run(args):
    specification = plan()
    runs = specification["jobs"]
    assignment_file = ROOT / "job-assignment.json"
    if assignment_file.exists():
        assignment = json.loads(assignment_file.read_bytes())
        allowed = {identity(job) for job in assignment["jobs"]}
        assert len(allowed) == len(assignment["jobs"])
        assert allowed <= {identity(job) for job in runs}
        runs = [job for job in runs if identity(job) in allowed]
    total = len(runs)
    active, failed, completed = {}, [], []
    status_file = ROOT / "status.json"
    if status_file.exists():
        previous = json.loads(status_file.read_bytes())
        for record in previous["active"]:
            job = {k: record[k] for k in ("recipe", "fold", "seed")}
            pid = record["pid"]
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                if identity(job) not in {identity(item) for item in runs}:
                    continue
                if b"--recipe" not in command or job["recipe"].encode() not in command:
                    continue
                key = f"{job['recipe']}-f{job['fold']}-s{job['seed']}"
                active[key] = AdoptedProcess(pid, job), job
            except (FileNotFoundError, ProcessLookupError):
                continue
        running = {identity(job) for _, job in active.values()}
        runs = [job for job in runs if identity(job) not in running]
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    while runs or active:
        for key, (process, job) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            if code:
                failed.append({**job, "exit_code": code})
            else:
                completed.append(job)
            del active[key]
        slots_file = ROOT / "slots"
        slots = int(slots_file.read_text().strip()) if slots_file.exists() else args.slots
        if not 1 <= slots <= 24:
            raise ValueError("Parallelism must be between 1 and 24")
        while runs and len(active) < slots:
            if (ROOT / "GLOBAL_PRIMARY_BARRIER").exists() and runs[0]["seed"] != 42:
                if not (ROOT / "PRIMARY_PHASE_RELEASED").exists():
                    break
            job = runs.pop(0)
            key = f"{job['recipe']}-f{job['fold']}-s{job['seed']}"
            directory = ROOT / "runs" / job["recipe"] / f"f{job['fold']}-s{job['seed']}"
            if (directory / "COMPLETE").exists():
                completed.append(job)
                continue
            command = [sys.executable, "-u", str(ROOT / "train_job.py"),
                       "--recipe", job["recipe"], "--fold", str(job["fold"]),
                       "--seed", str(job["seed"]), "--threads", str(args.threads)]
            with (logs / f"{key}.log").open("ab") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": str(args.threads),
                                  "MKL_NUM_THREADS": str(args.threads)})
            active[key] = process, job
        status = {
            "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "planned": total, "global_planned": 160, "completed": len(completed), "queued": len(runs),
            "active": [{"pid": p.pid, **j} for p, j in active.values()], "failed": failed,
        }
        write_json(ROOT / "status.json", status)
        time.sleep(3)
    if failed:
        raise SystemExit(f"{len(failed)} failed jobs, see status.json and logs")
    (ROOT / "TRAINING_COMPLETE").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, choices=range(1, 25), default=8)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    with (ROOT / "schedule.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
