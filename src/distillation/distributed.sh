#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source "${DISTILL_VENV:-/root/autodl-tmp/subpipe-raw-contrast/.venv}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u schedule.py --slots "${1:-16}" --threads 1
test -f TRAINING_COMPLETE
tar --exclude=resume.pt -czf shard-results.tar.gz.part runs status.json job-assignment.json
mv shard-results.tar.gz.part shard-results.tar.gz
sha256sum shard-results.tar.gz
if [[ "${2:-}" == main ]]; then
    while [[ $(find runs -name COMPLETE -type f | wc -l) -lt 160 ]]; do sleep 10; done
    python -u report.py
    tar --exclude=resume.pt --exclude='*.tmp' -czf distillation-results.tar.gz.part \
        runs profiles plan.json status.json audit.json experiment.py train_job.py objectives.py report.py \
        runs.csv summary.csv history.csv images.csv weights.csv calibration.json \
        original schedule.py calibrate.py preflight.py distributed.sh assets/weight-sha256.json \
        assets/reference.json assets/original-splits.csv assets/gray-strata.csv
    mv distillation-results.tar.gz.part distillation-results.tar.gz
    sha256sum distillation-results.tar.gz
fi
