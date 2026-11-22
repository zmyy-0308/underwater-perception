"""Export run-level metrics, seed/fold summaries and same-cohort paired changes."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multitask.engine import read


def write(path, rows):
    if not rows:
        return
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def compact(item, path):
    group = item.get("group") or {"task": "depth", "name": "baseline", "width": None}
    out = {"task": group["task"], "group": group["name"], "width": group["width"], "stage": item["stage"],
           "fold": item["fold"], "seed": item["seed"], "file": str(path)}
    sem, depth = item.get("semantic", {}), item.get("depth", {})
    for key in ("primary_iou", "miou", "boundary_f1", "no_fish_false_rate"):
        out[key] = sem.get(key)
    for key in ("mae_m", "rmse_m", "abs_rel", "valid_coverage", "gt_raster_coverage"):
        out[key] = depth.get(key)
    for name, values in sem.get("per_class", {}).items():
        out.update({f"{name}_{key}": value for key, value in values.items()})
    return out


def summarize(rows):
    groups, folds = defaultdict(list), defaultdict(list)
    for r in rows:
        group = (r["stage"], r["task"], r["group"], r["width"])
        groups[group].append(r)
        folds[group+(r["fold"],)].append(r)
    output = []
    for key, data in groups.items():
        row = dict(zip(("stage", "task", "group", "width"), key))
        row.update(runs=len(data), folds=len({r["fold"] for r in data}), seeds=len({r["seed"] for r in data}))
        for metric in ("primary_iou", "miou", "mae_m", "rmse_m", "abs_rel"):
            by_fold, variances = [], []
            for signature, observations in folds.items():
                if signature[:4] != key:
                    continue
                valid = [r[metric] for r in observations if r[metric] is not None]
                if valid:
                    by_fold.append(statistics.mean(valid))
                if len(valid) > 1:
                    variances.append(statistics.variance(valid))
            row[f"{metric}_mean"] = statistics.mean(by_fold) if by_fold else None
            row[f"{metric}_fold_sd"] = statistics.stdev(by_fold) if len(by_fold) > 1 else None
            row[f"{metric}_within_fold_seed_sd"] = statistics.mean(variances)**.5 if variances else None
        output.append(row)
    return output


def paired(records):
    reference = {}
    for item, row in records:
        if row["stage"] == "seg":
            key = (row["task"], row["group"], row["width"], row["fold"], row["seed"],
                   item.get("semantic", {}).get("signature"))
            if key in reference:
                raise ValueError(f"Duplicate semantic baseline for {key}")
            reference[key] = row
    depth = {item["depth"]["signature"]: row for item, row in records if row["stage"] == "depth" and "depth" in item}
    result = []
    for item, row in records:
        if row["stage"] != "joint":
            continue
        key = (row["task"], row["group"], row["width"], row["fold"], row["seed"],
               item.get("semantic", {}).get("signature"))
        base = reference.get(key)
        baseline = depth.get(item.get("depth", {}).get("signature"))
        comparison = {k: row[k] for k in ("task", "group", "width", "fold", "seed")}
        for metric, control in (("primary_iou", base), ("miou", base), ("mae_m", baseline), ("rmse_m", baseline)):
            comparison[f"delta_{metric}"] = (row[metric]-control[metric]
                                             if control and None not in (row[metric], control[metric]) else None)
        result.append(comparison)
    return result


def structural(records):
    pipe = (
        ["P5 P1", "P6 P2", "P7 P3", "P8 P4"],
        ["P3 P1", "P4 P2", "P7 P5", "P8 P6"],
        ["P2 P1", "P4 P3", "P6 P5", "P8 P7"],
    )
    bio = (["B3 B1", "B4 B2"], ["B2 B1", "B4 B3"], ["B5 B4"])
    unet = tuple([f"U{i+1} U{i+1-bit}" for i in range(8) if i & bit] for bit in (4, 1, 2))
    cohort = defaultdict(dict)
    for item, row in records:
        if row["stage"] == "seg" and "semantic" in item:
            key = (row["task"], row["fold"], row["seed"], item["semantic"]["signature"])
            label = (row["group"], row["width"])
            if label in cohort[key]:
                raise ValueError(f"Duplicate segmentation result {key}/{label}")
            cohort[key][label] = row
    results = []
    for (task, fold, seed, signature), rows in cohort.items():
        plan = []
        for family, factors in ((task, pipe if task == "pipe" else bio), ("unet", unet)):
            for name, terms in zip(("truncation", "dw" if family != "unet" else "haar",
                                    "haar" if family != "unet" else "attention"), factors):
                plan.extend((f"{family}_{name}", a, b) for a, b in map(str.split, terms))
        plan.extend([("wavelet_attention", "P9", "P8"), ("equal_parameter_spatial", "P10", "P8")]
                    if task == "pipe" else [("wavelet_attention", "B6", "B5"), ("attention_vs_none", "B6", "B4")])
        for w in sorted({w for _, w in rows}):
            for name, a, b in plan:
                if (a, w) in rows and (b, w) in rows:
                    results.append(semantic_delta(task, fold, seed, name, rows[a, w], rows[b, w]))
        for (g, w), row in rows.items():
            default = {"E2": 256, "E3": 128, "E4": 64, "E5": 128}.get(
                g, 256 if g.startswith("U") else 128 if task == "pipe" else 64)
            if default == 256 and (g, default) not in rows:
                default = 192
            if w != default and (g, default) in rows:
                results.append(semantic_delta(task, fold, seed, "width_vs_reference", row, rows[g, default]))
    return results


def semantic_delta(task, fold, seed, contrast, a, b):
    row = dict(task=task, fold=fold, seed=seed, contrast=contrast,
               model=f"{a['group']}-W{a['width']}", reference=f"{b['group']}-W{b['width']}")
    for field in ("primary_iou", "miou"):
        row[f"delta_{field}"] = a[field]-b[field] if None not in (a[field], b[field]) else None
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    records = []
    for path in sorted(args.results.rglob("summary.json")):
        item = read(path)
        if item.get("stage") in ("seg", "joint", "depth"):
            records.append((item, compact(item, path)))
    if not records:
        p.error("No multi-task evaluation summaries found")
    rows = [row for _, row in records]
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out/"runs.csv", rows)
    write(args.out/"groups.csv", summarize(rows))
    write(args.out/"joint_deltas.csv", paired(records))
    write(args.out/"structural_deltas.csv", structural(records))


if __name__ == "__main__":
    main()
