"""Connect the SubPipeMini and SUIM release packages to joint_run.py."""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_manifest(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def prepare_pipe(root, output):
    cache = read_json(root / "cache_288x512" / "meta.json")["splits"]
    files = {}
    for split in ("train", "val"):
        folder = root / split
        for stem in cache[split]["stems"]:
            image = next(path for path in (folder / (stem + ".jpg"), folder / (stem + ".png")) if path.is_file())
            mask = folder / (stem + "_label.png")
            with Image.open(image) as rgb, Image.open(mask) as label:
                if rgb.size != label.size:
                    raise ValueError(f"Unaligned SubPipeMini pair: {stem}")
            files[stem] = (image, mask)
    stems = sorted(files, key=float)
    protocol = read_json(root / "fivefold_splits.json")
    if len(stems) != 647 or protocol["purge_seconds"] != 2 or len(protocol["folds"]) != 5:
        raise ValueError("Unexpected SubPipeMini release inventory")
    expected = set(range(len(stems)))
    all_heldout = []
    rows = []
    for group in protocol["folds"]:
        fold, fit, test = group["fold"], group["train_indices"], group["test_indices"]
        if sorted(test) != list(range(min(test), max(test) + 1)):
            raise ValueError(f"Non-contiguous holdout in fold {fold}")
        lower, upper = float(stems[min(test)]), float(stems[max(test)])
        permitted_fit = {i for i in expected if float(stems[i]) < lower - 2 or float(stems[i]) > upper + 2}
        if len(fit) != len(set(fit)) or set(fit) != permitted_fit:
            raise ValueError(f"Invalid two-second purge in fold {fold}")
        all_heldout.extend(test)
        for split, selected in (("train", fit), ("test", test)):
            for idx in selected:
                stem = stems[idx]
                rows.append((f"fold{fold}:{stem}", *files[stem], "binary", split, fold))
        print(f"SubPipeMini fold {fold}: train={len(fit)}, test={len(test)}")
    if sorted(all_heldout) != sorted(expected):
        raise ValueError("The five holdouts must partition SubPipeMini")
    write_manifest(output / "subpipe-fivefold.csv", ("id", "image", "mask", "encoding", "split", "fold"), rows)


def prepare_suim(source, cache_root, output):
    splits = read_json(cache_root / "meta.json")["splits"]
    if [len(splits[key]["stems"]) for key in ("train", "val")] != [1369, 153]:
        raise ValueError("Unexpected SUIM development cache")
    test = source / "TEST"
    images = {path.stem: path for path in (test / "images").iterdir() if path.is_file()}
    labels = {path.stem: path for path in (test / "masks").iterdir() if path.is_file()}
    if len(images) != 108 or images.keys() - labels.keys():
        raise ValueError("Unexpected SUIM TEST pairs")
    rows = []
    for stem, image in sorted(images.items()):
        with Image.open(image) as rgb, Image.open(labels[stem]) as mask:
            if rgb.size != mask.size:
                raise ValueError(f"Unaligned SUIM TEST pair: {stem}")
        rows.append((stem, image, labels[stem], "rgb", "test"))
    write_manifest(output / "suim-test.csv", ("id", "image", "mask", "encoding", "split"), rows)
    print(f"SUIM train=1369, val=153, test={len(rows)}, unmatched masks={len(labels.keys() - images.keys())}")
    print(f"SUIM development source: {cache_root}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--release-root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    root = args.release_root.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=True)
    prepare_pipe(root / "SubPipeMini/subpipe", args.out)
    prepare_suim(root / "SUIM/source/suim", root / "SUIM/cache-and-labels/suim_clean/cache_384", args.out)


if __name__ == "__main__":
    main()
