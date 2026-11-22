"""Verify and merge the original fold and the eight follow-up trials."""

import argparse
import csv
import hashlib
import json
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path


STAGING = Path(__file__).resolve().parent / "attention-fivefold"
RESULT_ARCHIVE = "attention-fivefold-results.tar.gz"
PILOT_ARCHIVE = "stage1-attention-pilot.tar.gz"
ARCHITECTURES = {"P7": "无注意力", "P8": "Haar 小波", "P11": "ECA", "P12": "轻量 Swin"}
SCORES = ("primary_iou", "miou", "boundary_f1")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(filename, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (STAGING / filename).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_member(archive, filename):
    with archive.extractfile(filename) as stream:
        return stream.read()


def basic_row(group, fold, width=128):
    extra = {"P13": "轻量 Swin＋普通卷积", "P14": "坐标注意力", "P15": "条带深度卷积",
             "C1": "原图＋Canny", "C2": "原图＋局部灰度差", "C3": "原图＋Canny＋局部灰度差",
             "P5": "普通卷积", "C5": "普通卷积＋局部灰度差", "C6": "DW＋三子带 Haar＋局部灰度差"}
    label = extra.get(group, ARCHITECTURES.get(group))
    if group.startswith("H"):
        mask = int(group[1:])
        kept = [band for i, band in enumerate(("LL", "LH", "HL", "HH")) if mask & (1 << i)]
        label = "Haar 保留 " + "＋".join(kept)
    return {"group": group, "architecture": label, "fold": fold, "width": width, "seed": 42}


def read_baselines(desktop):
    rows = defaultdict(dict)
    source = read_csv(desktop / "01_逐次训练与评估汇总.csv")
    for item in source:
        group = item["group"]
        if (group not in ("P7", "P8") or item["task"] != "pipe"
                or item["seed"] != "42" or item["width"] != "128"):
            continue
        fold = int(item["fold"])
        assert fold not in rows[group]
        rows[group][fold] = {
            **basic_row(group, fold), **{k: float(item[k]) for k in SCORES},
            "test_images": int(item["test_images"]), "gpu": item["training_gpu"],
            "training_seconds": float(item["training_seconds"]),
            "peak_training_cuda_bytes": int(item["peak_training_cuda_bytes"]),
        }
    assert {k: sorted(v) for k, v in rows.items()} == {"P7": list(range(5)), "P8": list(range(5))}
    return rows


def read_run(archive, archive_name, folder, gpu, fold_reference, width=128):
    def data(name):
        return json.loads(read_member(archive, f"{folder}/{name}"))

    config = data("config.json")
    group = config["group"]["name"]
    fold = config["fold"]
    assert config["group"] == {"task": "pipe", "name": group, "width": width}
    assert group in ("P5", "P7", "P8", "P11", "P12", "P13", "P14", "P15", "C1", "C2", "C3", "C5", "C6") or (
        group.startswith("H") and group[1:].isdigit() and 1 <= int(group[1:]) < 15
    )
    reference = fold_reference[fold]
    for key in ("seed", "epochs", "batch", "effective_batch", "lr", "min_lr", "initial_sha256"):
        assert str(config[key]) == reference[key], (folder, key)
    evaluation = data("evaluation/summary.json")
    semantic = evaluation["semantic"]
    assert evaluation["group"] == config["group"]
    assert evaluation["fold"] == fold and evaluation["seed"] == 42
    assert evaluation["semantic_split"] == "test"
    assert semantic["images"] == int(reference["images"])
    histories = data("history.json")
    assert [r["epoch"] for r in histories] == list(range(1, 61))
    model_path = f"{folder}/model.pt"
    weight_data = read_member(archive, model_path)
    digest = hashlib.sha256(weight_data).hexdigest()
    assert digest == evaluation["checkpoint_sha256"]
    prefix = basic_row(group, fold, width)
    images = []
    for line in read_member(archive, f"{folder}/evaluation/predictions.jsonl").splitlines():
        item = json.loads(line)
        matrix = item["confusion"]
        boundary = item["boundary_counts"]
        out = {
            **prefix, "image_id": item["id"], "valid_px": item["valid_px"],
            "true_0_pred_0": matrix[0][0], "true_0_pred_1": matrix[0][1],
            "true_1_pred_0": matrix[1][0], "true_1_pred_1": matrix[1][1],
            "matched_pred_boundary": boundary[0], "predicted_boundary": boundary[1],
            "matched_truth_boundary": boundary[2], "truth_boundary": boundary[3],
        }
        assert sum(map(sum, matrix)) == item["valid_px"]
        images.append(out)
    assert len(images) == len({x["image_id"] for x in images}) == semantic["images"]
    tp = sum(row["true_1_pred_1"] for row in images)
    union = sum(row["true_0_pred_1"] + row["true_1_pred_0"] + row["true_1_pred_1"] for row in images)
    assert abs(tp / union - semantic["primary_iou"]) < 1e-12
    row = {
        **prefix, **{k: semantic[k] for k in SCORES},
        "test_images": semantic["images"], "gpu": gpu,
        "training_seconds": sum(x["seconds"] for x in histories),
        "peak_training_cuda_bytes": max(x["peak_cuda_bytes"] for x in histories),
    }
    return (
        row, [{**prefix, **x} for x in histories], images,
        {**prefix, "archive": archive_name, "weight_path": model_path,
         "sha256": digest, "weight_bytes": len(weight_data)},
        config["code_sha256"],
    )


def fold_members(path):
    expected = defaultdict(set)
    for row in read_csv(path):
        if row["split"] == "test":
            expected[int(row["fold"])].add(row["id"])
    return expected


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--desktop-results", type=Path, required=True)
    cmd.add_argument("--sha256", required=True)
    args = cmd.parse_args()
    target = STAGING / RESULT_ARCHIVE
    assert hashlib.sha256(target.read_bytes()).hexdigest() == args.sha256.strip().lower()
    references = {int(r["fold"]): r for r in read_csv(STAGING / "fold-reference.csv")}
    expected_frames = fold_members(args.desktop_results / "14_SubPipeMini五折划分.csv")
    rows = read_baselines(args.desktop_results)
    previous = {r["group"]: r for r in read_csv(
        args.desktop_results / "模型权重与补充实验/补充实验_模型对比.csv"
    )}
    epochs, images, weights = [], [], []
    revisions = set()
    for name, folds in ((PILOT_ARCHIVE, (0,)), (RESULT_ARCHIVE, range(1, 5))):
        path = STAGING / ("pilot.tar.gz" if name == PILOT_ARCHIVE else name)
        with tarfile.open(path, "r:gz") as archive:
            if name == PILOT_ARCHIVE:
                gpu, root = "RTX 5090", "attention-pilot-20261121"
            else:
                gpu = next(csv.DictReader(read_member(archive, "gpu.csv").decode().splitlines()))["name"]
                root = "runs"
                archived_reference = read_member(archive, "staging/fold-reference.csv")
                assert archived_reference == (STAGING / "fold-reference.csv").read_bytes()
            for group in ("P11", "P12"):
                for fold in folds:
                    folder = f"{root}/{group}-W128-f{fold}-s42"
                    row, history, item_rows, weight, revision = read_run(
                        archive, name, folder, gpu, references
                    )
                    assert row["fold"] == fold and row["group"] == group
                    assert {r["image_id"] for r in item_rows} == expected_frames[fold]
                    rows[group][fold] = row
                    epochs.extend(history)
                    images.extend(item_rows)
                    weights.append(weight)
                    revisions.add(revision)
    assert len(revisions) == 1, revisions
    assert (len(epochs), len(images), len(weights)) == (600, 1294, 10)

    all_folds, summaries = [], []
    for group in ARCHITECTURES:
        model = rows[group]
        assert set(model) == set(range(5))
        for fold, row in sorted(model.items()):
            row["parameters"] = int(previous[group]["parameters"])
            for base in ("P7", "P8"):
                row[f"pipe_iou_delta_vs_{base}_pp"] = (
                    row["primary_iou"] - rows[base][fold]["primary_iou"]
                ) * 100
            all_folds.append(row)
        summary = {
            "group": group, "architecture": ARCHITECTURES[group], "width": 128, "seed": 42,
            "folds": len(model), "test_images": sum(r["test_images"] for r in model.values()),
            "parameters": int(previous[group]["parameters"]),
        }
        for score in SCORES:
            values = [row[score] * 100 for row in model.values()]
            summary.update({f"{score}_mean_pct": statistics.mean(values),
                            f"{score}_fold_sd_pp": statistics.stdev(values),
                            f"{score}_min_pct": min(values), f"{score}_max_pct": max(values)})
        for base in ("P7", "P8"):
            changes = [r[f"pipe_iou_delta_vs_{base}_pp"] for r in model.values()]
            summary.update({f"mean_delta_vs_{base}_pp": statistics.mean(changes),
                            f"folds_improved_vs_{base}": sum(x > 0 for x in changes)})
        summary["RTX5090_fold0_fp16_model_p50_ms"] = previous[group]["fp16_model_p50_ms"]
        summaries.append(summary)
        print(group, f'IoU {summary["primary_iou_mean_pct"]:.3f} ± {summary["primary_iou_fold_sd_pp"]:.3f}%',
              "folds", [round(model[f]["primary_iou"] * 100, 2) for f in sorted(model)],
              f'vs Haar {summary["mean_delta_vs_P8_pp"]:+.3f} pp')
    write_csv("五折逐折对比.csv", all_folds)
    write_csv("五折指标汇总.csv", summaries)
    write_csv("补充实验_五折逐轮.csv", epochs)
    write_csv("补充实验_五折逐图.csv", images)
    write_csv("补充实验_五折权重索引.csv", weights)


if __name__ == "__main__":
    main()
