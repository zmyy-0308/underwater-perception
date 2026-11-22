#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec 9>gray-factorial.lock
flock -n 9
if [[ ! -f environment-ready ]]; then
    python -m venv --system-site-packages .venv
    .venv/bin/python -m pip --disable-pip-version-check install --no-index --no-deps staging/wheels/*.whl
    .venv/bin/python -c 'import torch, torchvision, cv2; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, cv2.__version__)'
    touch environment-ready
fi
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

code=attention-pilot-20261121/code
if [[ ! -d "$code" ]]; then
    tar -xzf gray-factorial-code.tar.gz "$code"
fi
groups=(P5 C5 P7 C2 H11 C6)
configs=()
for group in "${groups[@]}"; do configs+=("$group:64"); done
export EXPECTED_TRIALS=30 RESULT_ARCHIVE=gray-factorial-training.tar.gz
bash prepare_attention_fivefold.sh "${1:-20}" --configs "${configs[@]}" --folds 0 1 2 3 4
test "$(find runs -name COMPLETE -type f | wc -l)" -eq 30
(cd pipe-cache && sha256sum images.npy masks.npy) > dataset-cache.sha256

image=$(python -c 'import csv; print(next(r["image"] for r in csv.DictReader(open("manifests/subpipe-fivefold.csv")) if r["fold"]=="0" and r["split"]=="test"))')
for group in "${groups[@]}"; do
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
test "$(find profiles -type f -name timing.jsonl | wc -l)" -eq 12
tar --exclude='last.pt' -czf gray-factorial-results.tar.gz.part runs profiles gpu.csv dataset-cache.sha256 \
    gray-factorial-code.tar.gz run_attention_fivefold.py run_gray_factorial.sh \
    prepare_attention_fivefold.sh staging/fold-reference.csv manifests/fold-runtime-signatures.json
mv gray-factorial-results.tar.gz.part gray-factorial-results.tar.gz
sha256sum gray-factorial-results.tar.gz
