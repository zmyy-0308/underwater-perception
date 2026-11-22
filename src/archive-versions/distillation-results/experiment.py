"""Historical-model adapters, paired data, audited evaluation and experiment plan."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "original"))
from multitask.engine import restore_model, torchify_state, restore_state
from multitask.metrics import SegMetrics
from multitask.data import SemanticData
from objectives import RECIPES, REPEAT

ROLES = {"attention": "teacher", "plain": "plain", "haar": "student", "gray": "gray"}


def write_json(path, value):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save(path, value):
    temp = path.with_suffix(".pt.tmp")
    torch.save(value, temp)
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plan():
    jobs = [{"recipe": name, "fold": fold, "seed": seed}
            for seed in (42, 43, 44) for name in RECIPES
            if seed == 42 or name in REPEAT for fold in range(5)]
    assert len(RECIPES) == 24 and len(jobs) == 160
    payload = {"configs": 24, "main_runs": 120, "repeat_runs": 40, "jobs": jobs,
               "epochs": 60, "batch": 2, "effective_batch": 8, "input": [288, 512],
               "initialization": "matched historical student, followed by 60 fine-tuning epochs",
               "optimizer": "Adam", "lr": 1e-4, "min_lr": 1e-6, "weight_decay": 1e-4,
               "feature_weight": 0.1, "feature_temperature": 2,
               "tailoring": "binary adaptation: train-only per-batch PCA features and boundary target blending"}
    write_json(ROOT / "plan.json", payload)
    return payload


class Features(nn.Module):
    def __init__(self, model, teacher=False):
        super().__init__()
        self.model = model
        self.features = {}
        layers = {8: model.decode[1], 4: model.decode[2]} if teacher else {
            8: model.decoder.fuse8, 4: model.decoder.fuse4,
        }
        for scale, layer in layers.items():
            layer.register_forward_hook(self.tap(scale))

    def tap(self, scale):
        def capture(module, inputs, output):
            self.features[scale] = output
        return capture

    def forward(self, x):
        self.features.clear()
        logits = self.model(x)
        return logits, dict(self.features)


def load(role, fold, device):
    model, meta = restore_model(ROOT / "assets/weights" / f"{role}-f{fold}.pt", device)
    assert meta["fold"] == fold and meta["stage"] == "seg"
    return Features(model, teacher=role in ("teacher", "plain")), meta


class Samples(Dataset):
    def __init__(self, fold, split, seed=42, limit=0):
        source = ROOT / "data"
        metadata = json.loads((source / "pipe-cache/metadata.json").read_bytes())
        by_id = {Path(r["key"][0]).stem: i for i, r in enumerate(metadata["pairs"])}
        with (ROOT / "assets/original-splits.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["fold"]) == fold and r["split"] == split]
        self.rows = rows[:limit] if limit else rows
        self.indices = [by_id[Path(r["image"].replace("\\", "/")).stem] for r in self.rows]
        self.arrays = (
            np.load(source / "pipe-cache/images.npy", mmap_mode="r"),
            np.load(source / "pipe-cache/masks.npy", mmap_mode="r"),
        )
        with (ROOT / "assets/gray-strata.csv").open(encoding="utf-8-sig", newline="") as f:
            strata = {r["image_id"]: r["stratum"] for r in csv.DictReader(f)}
        self.strata = strata
        self.original = SemanticData.__new__(SemanticData)
        data = self.original
        data.task, data.size, data.augment, data.seed, data.epoch, data.canny = (
            "pipe", (288, 512), split == "train", seed, 0, False
        )
        data.arrays = self.arrays
        data.rows = [{**r, "index": i} for r, i in zip(self.rows, self.indices)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        x, y, image_id = self.original[index]
        return x, y, image_id

    def epoch(self, value):
        self.original.epoch = value


@torch.inference_mode()
def evaluate(model, dataset, device):
    model.eval()
    total = SegMetrics("pipe")
    groups = {name: SegMetrics("pipe") for name in ("low", "high")}
    group_counts = {name: 0 for name in groups}
    rows = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    for x, y, ids in loader:
        logits, _ = model(x.to(device))
        records = total.update(logits, y)
        for i, (image_id, record) in enumerate(zip(ids, records)):
            stratum = dataset.strata.get(image_id, "undefined")
            if stratum in groups:
                groups[stratum].update(logits[i:i+1], y[i:i+1])
                group_counts[stratum] += 1
            rows.append({"image_id": image_id, "stratum": stratum, **record})
    result = total.result()
    result["images"] = len(rows)
    result["strata"] = {name: {**counter.result(), "images": group_counts[name]}
                        for name, counter in groups.items()}
    return result, rows


def audit(device):
    references = json.loads((ROOT / "assets/reference.json").read_bytes())
    output = []
    for reference in references:
        fold, role = reference["fold"], reference["role"]
        model, meta = load(role, fold, device)
        test = Samples(fold, "test")
        result, _ = evaluate(model, test, device)
        error = abs(result["primary_iou"] - reference["iou"])
        row = {**reference, "retested_iou": result["primary_iou"], "absolute_error": error,
               "images": len(test)}
        output.append(row)
        write_json(ROOT / "audit.json", output)
        print("AUDIT", role, fold, result["primary_iou"], "error", error, flush=True)
        if error > 2e-5:
            raise ValueError(f"Historical forward mismatch: {role} fold {fold}")
        del model
    assert len(output) == 20 and sum(r["images"] for r in output if r["role"] == "teacher") == 647
    (ROOT / "AUDIT_OK").touch()
