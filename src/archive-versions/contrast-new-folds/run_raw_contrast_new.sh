#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec 9>raw-contrast.lock
flock -n 9
if [[ ! -f environment-ready ]]; then
    python -m venv --system-site-packages .venv
    .venv/bin/python -m pip --disable-pip-version-check install --no-index --no-deps staging/wheels/*.whl
    .venv/bin/python -c 'import torch, torchvision, cv2; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, cv2.__version__)'
    touch environment-ready
fi
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [[ ! -d attention-pilot-20261121/code ]]; then
    tar -xzf contrast-code.tar.gz attention-pilot-20261121/code
fi
export EXPECTED_TRIALS=12 RESULT_ARCHIVE=contrast-new-training.tar.gz
bash prepare_attention_fivefold.sh "${1:-12}" --configs P7:64 C1:64 C2:64 C3:64 --folds 0 1 2
test "$(find runs -name COMPLETE -type f | wc -l)" -eq 12
(cd pipe-cache && sha256sum images.npy masks.npy) > dataset-cache.sha256

code=attention-pilot-20261121/code
image=$(python -c 'import csv; print(next(r["image"] for r in csv.DictReader(open("manifests/subpipe-fivefold.csv")) if r["fold"]=="0" and r["split"]=="test"))')
for group in P7 C1 C2 C3; do
    tag="$group-W64"
    for precision in fp32 fp16; do
        out="profiles/$tag/$precision"
        [[ ! -f "$out/summary.json" ]] || continue
        flag=()
        if [[ "$precision" == fp16 ]]; then flag=(--fp16); fi
        python -u "$code/tools/joint_profile.py" --checkpoints "runs/$tag-f0-s42/model.pt" \
            --image "$image" --device cuda:0 --threads 2 --out "$out" "${flag[@]}"
    done
done
test "$(find profiles -type f -name timing.jsonl | wc -l)" -eq 8
tar --exclude='last.pt' -czf contrast-new-folds.tar.gz.part runs profiles gpu.csv dataset-cache.sha256 \
    contrast-code.tar.gz run_attention_fivefold.py run_raw_contrast_new.sh
mv contrast-new-folds.tar.gz.part contrast-new-folds.tar.gz
sha256sum contrast-new-folds.tar.gz
