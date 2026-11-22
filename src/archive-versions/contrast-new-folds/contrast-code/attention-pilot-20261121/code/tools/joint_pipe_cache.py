"""Cache the deterministic SubPipeMini preprocessing and reuse the training CLI."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joint_run
from multitask.data import SemanticData
from multitask.models import SIZES


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_stamp(path):
    info = path.stat()
    return [info.st_size, info.st_mtime_ns]


def pair_key(row):
    encoding = row.get("encoding") or "binary"
    threshold = int(row.get("threshold") or 0) if encoding == "binary" else 0
    return [str(row["image"]), str(row["mask"]), encoding, threshold]


def build(manifest, target):
    datasets = [SemanticData("pipe", manifest, "test", fold=fold) for fold in range(5)]
    total = sum(map(len, datasets))
    height, width = SIZES["pipe"]
    if total != len({tuple(pair_key(row)) for data in datasets for row in data.rows}) or total != 647:
        raise ValueError("Expected five disjoint test folds covering the 647 frames")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    with tempfile.TemporaryDirectory(prefix=".pipe-cache-", dir=target.parent) as temp:
        stage = Path(temp) / "data"
        stage.mkdir()
        images = np.lib.format.open_memmap(stage/"images.npy", "w+", np.uint8, (total, height, width, 3))
        masks = np.lib.format.open_memmap(stage/"masks.npy", "w+", np.uint8, (total, height, width))
        records = []
        for data in datasets:
            for n, row in enumerate(data.rows):
                im, mask, _ = data[n]
                index = len(records)
                images[index] = torch.round(im.permute(1, 2, 0)*255).to(torch.uint8).numpy()
                masks[index] = mask.numpy()
                records.append({"key": pair_key(row), "stamps": [file_stamp(row[k]) for k in ("image", "mask")]})
        images.flush()
        masks.flush()
        del images, masks
        (stage/"metadata.json").write_text(json.dumps({
            "manifest_sha256": fingerprint(manifest), "code_sha256": joint_run.code_digest(), "pairs": records,
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        stage.rename(target)
    print(f"Cached {total} aligned image/mask pairs at {target}", flush=True)


def cached_dataset(cache):
    metadata = json.loads((cache/"metadata.json").read_text(encoding="utf-8"))
    if metadata["code_sha256"] != joint_run.code_digest():
        raise ValueError("The cache was created by another preprocessing revision")
    pairs = {tuple(row["key"]): (i, row["stamps"]) for i, row in enumerate(metadata["pairs"])}
    images = np.load(cache/"images.npy", mmap_mode="r")
    masks = np.load(cache/"masks.npy", mmap_mode="r")
    height, width = SIZES["pipe"]
    if (images.shape != (len(pairs), height, width, 3)
            or masks.shape != (len(pairs), height, width)
            or images.dtype != np.uint8 or masks.dtype != np.uint8):
        raise ValueError("Invalid pipe cache arrays")

    class CachedSemanticData(SemanticData):
        def __init__(self, task, path, *args, **kwargs):
            super().__init__(task, path, *args, **kwargs)
            if task != "pipe":
                return
            if fingerprint(Path(path)) != metadata["manifest_sha256"]:
                raise ValueError("The pipe manifest differs from the cache source")
            for row in self.rows:
                index, stamps = pairs[tuple(pair_key(row))]
                if [file_stamp(row[k]) for k in ("image", "mask")] != stamps:
                    raise ValueError(f"The cache source changed: {row['id']}")
                row["index"] = index
            self.arrays = (images, masks)
            print("Using verified pipe cache", cache, flush=True)

    return CachedSemanticData


def verify(manifest, cache):
    fast_dataset = cached_dataset(cache)
    checked = 0
    for fold in range(5):
        for split, augment in (("train", True), ("test", False)):
            args = ("pipe", manifest, split, fold)
            original = SemanticData(*args, limit=8, augment=augment, seed=43)
            cached = fast_dataset(*args, limit=8, augment=augment, seed=43)
            original.epoch = cached.epoch = 7
            assert original.signature == cached.signature
            for index in range(len(original)):
                x, y, name = original[index]
                a, b, identifier = cached[index]
                if name != identifier or not torch.equal(x, a) or not torch.equal(y, b):
                    raise ValueError(f"Cache changed a training/evaluation tensor: {name}")
                checked += 1
    print(f"Exact images, masks and IDs checked on {checked} fold samples", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify", "run"))
    parser.add_argument("cache", type=Path)
    args, rest = parser.parse_known_args()
    if args.mode == "run":
        joint_run.SemanticData = cached_dataset(args.cache.resolve())
        sys.argv = [sys.argv[0], *rest]
        joint_run.main()
    else:
        if len(rest) != 1:
            parser.error("Supply exactly one SubPipeMini manifest")
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        (build if args.mode == "build" else verify)(Path(rest[0]).resolve(), args.cache.resolve())


if __name__ == "__main__":
    main()
