"""Restricted SSH commands for the three-card experiment coordinator."""
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def jobs():
    return json.loads((ROOT / "job-assignment.json").read_bytes())["jobs"]


def folder(job):
    return ROOT / "runs" / job["recipe"] / f"f{job['fold']}-s{job['seed']}"


def status():
    assigned = jobs()
    done = [j for j in assigned if (folder(j) / "COMPLETE").exists()]
    return {
        "planned": len(assigned), "completed": len(done),
        "primary_planned": sum(j["seed"] == 42 for j in assigned),
        "primary_done": sum(j["seed"] == 42 for j in done),
    }


def bundle(primary=False):
    assigned = [j for j in jobs() if not primary or j["seed"] == 42]
    assert all((folder(j) / "COMPLETE").exists() for j in assigned)
    archive = ROOT / ("primary-shard.tar.gz" if primary else "final-shard.tar.gz")
    if not archive.exists():
        temp = archive.with_suffix(".gz.tmp")
        with tarfile.open(temp, "w:gz", compresslevel=1) as output:
            for job in assigned:
                for filename in ("COMPLETE", "model.pt", "config.json", "history.json",
                                 "summary.json", "predictions.jsonl"):
                    path = folder(job) / filename
                    output.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        temp.replace(archive)
    return archive


def serve(command):
    if command == "status":
        print(json.dumps(status()))
    elif command == "release":
        (ROOT / "PRIMARY_PHASE_RELEASED").touch()
        print("released")
    elif command in ("primary", "final"):
        path = bundle(command == "primary")
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                sys.stdout.buffer.write(block)
    else:
        raise SystemExit("Allowed experiment commands: status, release, primary, final")


if __name__ == "__main__":
    serve(os.environ.get("SSH_ORIGINAL_COMMAND", "").strip())
