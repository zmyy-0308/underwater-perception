"""Collect verified historical weights and the frozen experiment protocol."""
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV = Path.home() / "Desktop/水下感知第一阶段实验数据_CSV"
BASE = CSV / "模型权重与补充实验"
OUT = ROOT / "assets"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    OUT.mkdir(exist_ok=True)
    weights = OUT / "weights"
    weights.mkdir(exist_ok=True)
    requested = {}
    for row in read(BASE / "00_权重索引.csv"):
        parts = row["run_id"].split("/")
        if parts[:2] == ["seg", "pipe"] and parts[2] in ("U1-W128", "U3-W128") and parts[3].endswith("s42"):
            role = "teacher" if parts[2] == "U3-W128" else "plain"
            fold = int(parts[3].split("-")[0][1:])
            requested.setdefault(BASE / row["archive"], []).append(
                (row["path"], weights / f"{role}-f{fold}.pt", row["sha256"])
            )
    origin = BASE / "灰度差与解码器消融"
    for row in read(origin / "灰度差与解码器_权重索引.csv"):
        if row["group"] in ("H11", "C6") and row["seed"] == "42":
            role = "student" if row["group"] == "H11" else "gray"
            requested.setdefault(origin / row["archive"], []).append(
                (row["weight_path"], weights / f"{role}-f{row['fold']}.pt", row["sha256"])
            )
    manifest = []
    for archive, entries in requested.items():
        with tarfile.open(archive, "r:*") as source:
            for member, destination, expected in entries:
                if not destination.exists():
                    with source.extractfile(member) as stream:
                        content = stream.read()
                    assert hashlib.sha256(content).hexdigest() == expected
                    destination.write_bytes(content)
                assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected
                manifest.append({"file": destination.name, "sha256": expected})
    assert len(manifest) == 20 and len({r["file"] for r in manifest}) == 20
    shutil.copyfile(CSV / "14_SubPipeMini五折划分.csv", OUT / "original-splits.csv")
    strata = BASE / "Canny与局部灰度差/Canny与局部灰度差_逐图灰度分组.csv"
    shutil.copyfile(strata, OUT / "gray-strata.csv")
    reference = []
    for row in read(CSV / "01_逐次训练与评估汇总.csv"):
        if row["task"] == "pipe" and row["group"] in ("U1", "U3") and row["seed"] == "42":
            reference.append({"role": "teacher" if row["group"] == "U3" else "plain",
                              "fold": int(row["fold"]), "iou": float(row["primary_iou"])})
    for row in read(origin / "灰度差与解码器_逐折结果.csv"):
        if row["group"] in ("H11", "C6"):
            reference.append({"role": "student" if row["group"] == "H11" else "gray",
                              "fold": int(row["fold"]), "iou": float(row["primary_iou"])})
    assert len(reference) == 20
    (OUT / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    (OUT / "weight-sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(manifest)} weights and {len(reference)} matched reference metrics")


if __name__ == "__main__":
    main()
