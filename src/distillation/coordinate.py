"""Release repeat seeds after all 120 primaries and merge both remote shards."""
import csv
import json
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from bridge import status
from experiment import ROOT, digest, write_json

KEY = ROOT / "coordination-key"
PORTS = (32495, 31084)


def remote(port, command, destination=None):
    call = ["ssh", "-i", str(KEY), "-p", str(port), "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=12", "-o", f"UserKnownHostsFile={ROOT}/coordination-known-hosts",
            "-o", "StrictHostKeyChecking=yes", "root@connect.bjb2.seetacloud.com", command]
    for attempt in range(5):
        try:
            if destination:
                temp = destination.with_suffix(".tmp")
                with temp.open("wb") as output:
                    subprocess.run(call, stdout=output, check=True, timeout=600)
                temp.replace(destination)
                return
            return json.loads(subprocess.check_output(call, text=True, timeout=40)) if command == "status" else (
                subprocess.check_output(call, text=True, timeout=40)
            )
        except (subprocess.SubprocessError, OSError):
            if attempt == 4:
                raise
            time.sleep(10)


def merge(archive):
    with tarfile.open(archive, "r:gz") as source:
        for entry in source:
            path = ROOT / entry.name
            resolved = path.resolve()
            if not resolved.is_relative_to((ROOT / "runs").resolve()) or not entry.isfile():
                raise ValueError(f"Unexpected shard member {entry.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(entry) as stream:
                content = stream.read()
            if path.exists():
                if path.read_bytes() != content:
                    raise ValueError(f"Conflicting result {entry.name}")
                continue
            temp = path.with_suffix(path.suffix + ".merge")
            temp.write_bytes(content)
            temp.replace(path)


def primary_report():
    rows = []
    for path in ROOT.glob("runs/*/f*-s42/summary.json"):
        value = json.loads(path.read_bytes())
        rows.append({k: value[k] for k in ("recipe", "fold", "seed", "primary_iou", "boundary_f1")})
    assert len(rows) == 120
    with (ROOT / "primary-runs.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scores = {}
    for name in {r["recipe"] for r in rows}:
        group = [r for r in rows if r["recipe"] == name]
        assert len(group) == 5
        scores[name] = {"mean_iou_pct": statistics.mean(r["primary_iou"] * 100 for r in group),
                        "worst_iou_pct": min(r["primary_iou"] * 100 for r in group)}
    write_json(ROOT / "primary-summary.json", scores)


def main():
    merged = ROOT / "merged-shards"
    merged.mkdir(exist_ok=True)
    while True:
        states = [status(), *(remote(port, "status") for port in PORTS)]
        write_json(ROOT / "three-card-status.json", {"cards": states})
        assert sum(s["primary_planned"] for s in states) == 120
        assert sum(s["planned"] for s in states) == 160
        if all(s["primary_done"] == s["primary_planned"] for s in states):
            break
        time.sleep(20)
    for port in PORTS:
        archive = merged / f"primary-{port}.tar.gz"
        remote(port, "primary", archive)
        merge(archive)
    primary_report()
    (ROOT / "PRIMARY_PHASE_RELEASED").touch()
    for port in PORTS:
        remote(port, "release")
    print("All 120 primary evaluations merged, released 40 seed repeats", flush=True)
    while True:
        states = [status(), *(remote(port, "status") for port in PORTS)]
        write_json(ROOT / "three-card-status.json", {"cards": states})
        if all(s["completed"] == s["planned"] for s in states):
            break
        time.sleep(20)
    for port in PORTS:
        archive = merged / f"final-{port}.tar.gz"
        remote(port, "final", archive)
        merge(archive)
    assert len(list(ROOT.glob("runs/*/*/COMPLETE"))) == 160
    print("All 160 results merged; main pipeline continues profiling and packaging", flush=True)


if __name__ == "__main__":
    main()
