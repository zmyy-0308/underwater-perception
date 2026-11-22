#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec 9>schedule.lock
flock -n 9
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

dataset=
for attempt in $(seq 1 180); do
    for candidate in staging/subpipe.zip staging/subpipe-upload.zip; do
        if [[ -f "$candidate" && $(stat -c %s "$candidate") == 408857121 ]]; then
            dataset=$candidate
            break 2
        fi
    done
    sleep 2
done
test -n "$dataset"
printf '%s  %s\n' 49430a5f7f7c64937780abb5f65fcf226896ee1cb8c8e767db409815ebd04197 "$dataset" \
    7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934 staging/efficientnet_b0_rwightman-7f5810bc.pth \
    8d891fdc61a53f0b10eff22ebd460f93b6a85bd28f86dca66c314667ff60dc40 staging/pilot.tar.gz | sha256sum -c

if [[ ! -d attention-pilot-20261121/code ]]; then
    tar -xzf staging/pilot.tar.gz attention-pilot-20261121/code
fi
if [[ -f staging/ablation-models.py ]]; then
    install -m 0644 staging/ablation-models.py attention-pilot-20261121/code/multitask/models.py
fi
if [[ ! -f datasets/extracted ]]; then
    mkdir -p datasets
    unzip -oq "$dataset" -d datasets
    touch datasets/extracted
fi

export PYTHONPATH="$PWD/attention-pilot-20261121/code"
mkdir -p manifests
python -u - <<'PY'
import csv
import hashlib
import json
from pathlib import Path
import cv2
import torch
import torchvision
from multitask.data import SemanticData
from tools.prepare_released_segmentation import prepare_pipe

root = Path.cwd()
out = root / "manifests"
prepare_pipe(root / "datasets/subpipe", out)
runtime_signatures = {}
def identity(dataset):
    payload = json.dumps([[row["id"], row.get("encoding") or "binary"] for row in dataset.rows],
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()

with (root / "staging/fold-reference.csv").open(encoding="utf-8-sig", newline="") as source:
    for reference in csv.DictReader(source):
        fold = int(reference["fold"])
        manifest = out / "subpipe-fivefold.csv"
        train = SemanticData("pipe", manifest, "train", fold, 0, True, 42)
        test = SemanticData("pipe", manifest, "test", fold)
        if identity(train) != reference["train_ids_sha256"] or identity(test) != reference["test_ids_sha256"]:
            raise ValueError(f"Archived split membership or ordering does not match fold {fold}")
        if len(test) != int(reference["images"]):
            raise ValueError(f"Archived test size does not match fold {fold}")
        print(f"Verified fold {fold}: train={len(train)}, test={len(test)}")
        runtime_signatures[str(fold)] = {
            "semantic_signature": train.signature, "test_signature": test.signature,
        }
(out / "fold-runtime-signatures.json").write_text(json.dumps(runtime_signatures), encoding="utf-8")
print(torch.cuda.get_device_name(0), torch.__version__, torchvision.__version__, cv2.__version__)
PY

tool=attention-pilot-20261121/code/tools/joint_pipe_cache.py
if [[ ! -f pipe-cache/metadata.json ]]; then
    python -u "$tool" build pipe-cache manifests/subpipe-fivefold.csv
fi
python -u "$tool" verify pipe-cache manifests/subpipe-fivefold.csv
slots=${1:-2}
if [[ $# -gt 0 ]]; then shift; fi
python -u run_attention_fivefold.py --slots "$slots" "$@"
test "$(find runs -name COMPLETE -type f | wc -l)" -eq "${EXPECTED_TRIALS:-8}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > gpu.csv
archive=${RESULT_ARCHIVE:-attention-fivefold-results.tar.gz}
provenance=(runs gpu.csv staging/fold-reference.csv manifests/fold-runtime-signatures.json)
if [[ -f staging/ablation-models.py ]]; then
    provenance+=(staging/ablation-models.py run_attention_fivefold.py prepare_attention_fivefold.sh)
fi
tar --exclude='last.pt' -czf "$archive.part" "${provenance[@]}"
mv "$archive.part" "$archive"
sha256sum "$archive"
