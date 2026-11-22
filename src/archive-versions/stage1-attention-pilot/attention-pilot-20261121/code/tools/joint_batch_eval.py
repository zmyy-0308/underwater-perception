"""Evaluate a plan range without rebuilding cohorts from test results."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_run import evaluate_command, job_path, source_options
from multitask.engine import digest, json_write, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source_options(p)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--sem-split", choices=("val", "test"), default="test")
    p.add_argument("--stereo-part", choices=("val", "test"), default="test")
    args = p.parse_args()
    if min(args.count, args.threads) <= 0 or min(args.start, args.limit_sem, args.limit_stereo) < 0:
        p.error("Invalid job slice or runtime options")
    cv2.setNumThreads(args.threads)
    torch.set_num_threads(args.threads)
    jobs = read(args.plan)["jobs"][args.start:args.start+args.count]
    if not jobs:
        p.error("Empty plan slice")
    status = []
    for job in jobs:
        ckpt = job_path(args.runs, job)/"model.pt"
        out = job_path(args.out, job)
        try:
            if job["task"] == "pipe" and args.pipe_source is None:
                raise ValueError("Supply the corresponding pipe dataset")
            if job["task"] == "bio" and args.bio_source is None:
                raise ValueError("Supply the corresponding bio dataset")
            if job["stage"] != "seg" and args.stereo_zip is None:
                raise ValueError("Supply the FLSea archive")
            if (out/"summary.json").exists():
                previous = read(out/"summary.json")
                if previous["checkpoint_sha256"] != digest(ckpt):
                    raise ValueError("Existing evaluation belongs to another checkpoint")
                if previous["semantic_split"] not in (None, args.sem_split) or previous["stereo_split"] not in (None, args.stereo_part):
                    raise ValueError("Use another evaluation output for a different split")
            else:
                evaluate_command(SimpleNamespace(**{**vars(args), "checkpoint": ckpt, "out": out}))
            result = {"status": "complete", "job": job, "path": str(out)}
        except Exception as exc:
            result = {"status": "failed", "job": job, "error": f"{type(exc).__name__}: {exc}"}
        status.append(result)
        json_write(args.out/f"eval-{args.start}.json", status)
    if any(row["status"] == "failed" for row in status):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
