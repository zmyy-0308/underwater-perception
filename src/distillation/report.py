"""Collect paired metrics and serial deployment timing after training completes."""
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from experiment import ROOT, digest, plan, write_json


def read(path):
    return json.loads(path.read_bytes())


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    jobs = plan()["jobs"]
    results, histories, images, checkpoints = [], [], [], []
    for job in jobs:
        recipe, fold, seed = (job[k] for k in ("recipe", "fold", "seed"))
        out = ROOT / "runs" / recipe / f"f{fold}-s{seed}"
        assert (out / "COMPLETE").exists()
        result = read(out / "summary.json")
        assert result["checkpoint_sha256"] == digest(out / "model.pt")
        assert result["images"] == (130 if fold in (1, 3) else 129)
        row = {**job, **{k: v for k, v in result.items() if k not in ("strata", "per_class")}}
        for name, metrics in result["strata"].items():
            row.update({f"{name}_{k}": v for k, v in metrics.items() if k != "per_class"})
        results.append(row)
        hist = read(out / "history.json")
        assert len(hist) == 60 and hist[-1]["epoch"] == 60
        histories.extend({**job, **epoch} for epoch in hist)
        with (out / "predictions.jsonl").open() as source:
            frame_records = [json.loads(line) for line in source]
        assert len({r["image_id"] for r in frame_records}) == result["images"]
        for frame in frame_records:
            confusion = frame.pop("confusion")
            boundary = frame.pop("boundary_counts")
            images.append({
                **job, **frame, **{f"true_{i}_pred_{j}": v for i, line in enumerate(confusion) for j, v in enumerate(line)},
                **dict(zip(("matched_pred_boundary", "predicted_boundary", "matched_true_boundary", "true_boundary"), boundary)),
            })
        checkpoints.append({**job, "path": str((out / "model.pt").relative_to(ROOT)),
                            "sha256": result["checkpoint_sha256"], "bytes": (out / "model.pt").stat().st_size})
    assert len(results) == 160 and len(histories) == 9600 and len(images) == 20704
    baselines = {(r["fold"], r["seed"]): r for r in results if r["recipe"] == "supervised"}
    gray = {(r["fold"], r["seed"]): r for r in results if r["recipe"] == "gray_supervised"}
    for row in results:
        baseline = (gray if row["recipe"].startswith("gray") else baselines)[row["fold"], row["seed"]]
        for key in ("primary_iou", "boundary_f1", "low_primary_iou"):
            row[f"delta_{key}_pp"] = 100 * (row[key] - baseline[key])
    summaries = []
    for name in dict.fromkeys(r["recipe"] for r in results):
        group = [r for r in results if r["recipe"] == name]
        for seed in sorted({r["seed"] for r in group}):
            selected = [r for r in group if r["seed"] == seed]
            assert len(selected) == 5
            scores = [r["primary_iou"] * 100 for r in selected]
            summaries.append({
                "recipe": name, "seed": seed, "folds": 5,
                "mean_iou_pct": statistics.mean(scores), "fold_sd_pp": statistics.stdev(scores),
                "worst_iou_pct": min(scores),
                "mean_paired_iou_delta_pp": statistics.mean(r["delta_primary_iou_pp"] for r in selected),
                "mean_low_iou_pct": statistics.mean(r["low_primary_iou"] * 100 for r in selected),
                "mean_boundary_f1_pct": statistics.mean(r["boundary_f1"] * 100 for r in selected),
            })
    for name, rows in (("runs.csv", results), ("summary.csv", summaries), ("history.csv", histories),
                       ("images.csv", images), ("weights.csv", checkpoints)):
        csv_write(ROOT / name, rows)

    # Distillation heads are training-only. Profile both deployed student architectures serially.
    manifest = ROOT / "data/manifests/subpipe-fivefold.csv"
    with manifest.open() as f:
        image = next(r["image"] for r in csv.DictReader(f) if r["fold"] == "0" and r["split"] == "test")
    for recipe in ("supervised", "gray_supervised"):
        for precision in ("fp32", "fp16"):
            out = ROOT / "profiles" / recipe / precision
            if (out / "summary.json").exists():
                continue
            command = [sys.executable, str(ROOT / "original/tools/joint_profile.py"),
                       "--checkpoints", str(ROOT / "runs" / recipe / "f0-s42/model.pt"),
                       "--image", image, "--device", "cuda:0", "--threads", "2", "--out", str(out)]
            if precision == "fp16":
                command.append("--fp16")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
