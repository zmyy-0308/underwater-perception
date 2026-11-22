"""Resume the remaining attention ablations on one CUDA device."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "attention-pilot-20261121/code"
MANIFEST = ROOT / "manifests/subpipe-fivefold.csv"
CACHE = ROOT / "pipe-cache"
WEIGHT = ROOT / "staging/efficientnet_b0_rwightman-7f5810bc.pth"
OUTPUT = ROOT / "runs"
THREADS = 2
GROUPS = ("P7", "P11", "P12", "P13")
WIDTHS = (64, 96, 128)


def read_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def checked_config(run, group, fold, width, reference):
    config = read_json(run / "config.json")
    if config["group"] != {"width": width, "name": group, "task": "pipe"}:
        raise ValueError(f"Unexpected model configuration: {run}")
    for key in ("semantic_signature", "initial_sha256", "epochs", "batch",
                "effective_batch", "lr", "min_lr", "seed", "fold"):
        if str(config[key]) != reference[key]:
            raise ValueError(f"Configuration mismatch for {run}: {key}")


def run_trial(group, fold, width, reference):
    run = OUTPUT / f"{group}-W{width}-f{fold}-s42"
    model = run / "model.pt"
    evaluation = run / "evaluation"
    log = OUTPUT / f"{run.name}.log"
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS)}
    common = ["--device", "cuda:0", "--threads", str(THREADS), "--pipe-source", str(MANIFEST)]

    with log.open("a", encoding="utf-8") as stream:
        if not model.is_file():
            resume = ["--resume"] if (run / "last.pt").is_file() else []
            subprocess.run(
                [sys.executable, "-u", str(CODE / "tools/joint_pipe_cache.py"),
                 "run", str(CACHE), "train", "--stage", "seg", "--task", "pipe",
                 "--group", group, "--width", str(width), "--seed", "42", "--fold", str(fold),
                 "--epochs", "60", "--batch", "2", "--weight", str(WEIGHT),
                 "--out", str(run), *common, *resume],
                env=env, stdout=stream, stderr=subprocess.STDOUT, check=True,
            )
        checked_config(run, group, fold, width, reference)
        history = read_json(run / "history.json")
        if [item["epoch"] for item in history] != list(range(1, 61)):
            raise ValueError(f"Incomplete epoch history: {run}")
        if not (evaluation / "summary.json").is_file():
            subprocess.run(
                [sys.executable, "-u", str(CODE / "joint_run.py"), "eval",
                 "--checkpoint", str(model), "--out", str(evaluation),
                 "--sem-split", "test", *common],
                env=env, stdout=stream, stderr=subprocess.STDOUT, check=True,
            )
        summary = read_json(evaluation / "summary.json")
        if (summary["semantic"]["signature"] != reference["test_signature"]
                or summary["semantic"]["images"] != int(reference["images"])
                or summary["checkpoint_sha256"] != sha256(model)):
            raise ValueError(f"Evaluation and reference do not match: {run}")
        with (evaluation / "predictions.jsonl").open(encoding="utf-8") as predictions:
            if sum(1 for _ in predictions) != int(reference["images"]):
                raise ValueError(f"Incomplete predictions: {run}")
        (run / "COMPLETE").touch()
        return run.name, summary["semantic"]["primary_iou"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, choices=range(1, 21), default=2)
    parser.add_argument("--groups", choices=GROUPS, nargs="+", default=("P11", "P12"))
    parser.add_argument("--width", type=int, choices=WIDTHS, default=128)
    parser.add_argument("--configs", nargs="+", metavar="GROUP:WIDTH")
    parser.add_argument("--folds", type=int, choices=range(5), nargs="+", default=range(1, 5))
    args = parser.parse_args()
    configurations = [(group, args.width) for group in args.groups]
    if args.configs:
        try:
            configurations = [(group, int(width)) for group, width in (spec.split(":") for spec in args.configs)]
        except ValueError:
            parser.error("Supply each configuration as GROUP:WIDTH")
        if any(group not in GROUPS or width not in WIDTHS for group, width in configurations):
            parser.error("Unknown experiment group or decoder width")
    if len(set(configurations)) != len(configurations) or len(set(args.folds)) != len(args.folds):
        parser.error("Each configuration and fold can appear only once")
    with (ROOT / "staging/fold-reference.csv").open(encoding="utf-8-sig", newline="") as source:
        references = {int(row["fold"]): row for row in csv.DictReader(source)}
    if set(references) != set(range(5)):
        raise ValueError("Expected all five archived reference splits")
    runtime_signatures = read_json(ROOT / "manifests/fold-runtime-signatures.json")
    for fold, reference in references.items():
        reference.update(runtime_signatures[str(fold)])
    OUTPUT.mkdir(exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.slots) as pool:
        futures = {
            pool.submit(run_trial, group, fold, width, references[fold]): (group, width, fold)
            for fold in args.folds for group, width in configurations
        }
        for future in as_completed(futures):
            try:
                name, iou = future.result()
                print(f"COMPLETE {name} pipe_iou={iou:.8f}", flush=True)
            except Exception as error:
                job = futures[future]
                failures.append(job)
                print(f"FAILED {job}: {error}", file=sys.stderr, flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} trials require recovery")


if __name__ == "__main__":
    main()
