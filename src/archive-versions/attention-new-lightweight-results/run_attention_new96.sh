#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
parent=$(dirname "$PWD")
source "$parent/.venv/bin/activate"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec 9>new96.lock
flock -n 9

code=attention-pilot-20261121/code
if [[ ! -d "$code" ]]; then
    tar -xzf "$parent/staging/pilot.tar.gz" "$code"
fi
install -m 0644 models.py "$code/multitask/models.py"
install -m 0644 attention.py "$code/multitask/attention.py"
for directory in staging manifests; do
    [[ -e "$directory" ]] || ln -s "$parent/$directory" "$directory"
done
tool="$code/tools/joint_pipe_cache.py"
if [[ ! -f pipe-cache/metadata.json ]]; then
    python -u "$tool" build pipe-cache manifests/subpipe-fivefold.csv
fi
python -u "$tool" verify pipe-cache manifests/subpipe-fivefold.csv
(cd pipe-cache && sha256sum images.npy masks.npy) > dataset-cache.sha256
echo "Ready: P7-W96, P14-W96, P15-W96"

while [[ ! -f "$parent/attention-width96-profiles.tar.gz" ]]; do
    sleep 5
done
python -u run_attention_fivefold.py --slots 15 --configs P7:96 P14:96 P15:96 --folds 0 1 2 3 4
test "$(find runs -name COMPLETE -type f | wc -l)" -eq 15
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > gpu.csv

image=$(python -c 'import csv; print(next(r["image"] for r in csv.DictReader(open("manifests/subpipe-fivefold.csv")) if r["fold"]=="0" and r["split"]=="test"))')
for group in P7 P14 P15; do
    tag="$group-W96"
    for precision in fp32 fp16; do
        out="profiles/$tag/$precision"
        [[ ! -f "$out/summary.json" ]] || continue
        flag=()
        if [[ "$precision" == fp16 ]]; then flag=(--fp16); fi
        python -u "$code/tools/joint_profile.py" --checkpoints "runs/$tag-f0-s42/model.pt" \
            --image "$image" --device cuda:0 --threads 2 --out "$out" "${flag[@]}"
    done
done
test "$(find profiles -type f -name timing.jsonl | wc -l)" -eq 6
archive=attention-new-lightweight-results.tar.gz
tar --exclude='last.pt' -czf "$archive.part" runs profiles gpu.csv dataset-cache.sha256 \
    models.py attention.py run_attention_fivefold.py run_attention_new96.sh
mv "$archive.part" "$archive"
sha256sum "$archive"
