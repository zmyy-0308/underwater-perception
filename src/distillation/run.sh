#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /root/autodl-tmp/subpipe-raw-contrast/.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -f AUDIT_OK ]]; then
    python -u -c 'from experiment import audit, plan; import torch; torch.set_num_threads(2); plan(); audit(torch.device("cuda:0"))'
fi
if [[ ! -f PREFLIGHT_OK ]]; then python -u preflight.py; fi
if [[ ! -f chosen-slots ]]; then python -u calibrate.py; fi
slots=${1:-$(cat chosen-slots)}
python -u schedule.py --slots "$slots" --threads 1
python -u report.py
tar --exclude=resume.pt --exclude='*.tmp' -czf distillation-results.tar.gz.part \
    runs profiles plan.json status.json audit.json experiment.py train_job.py objectives.py report.py \
    runs.csv summary.csv history.csv images.csv weights.csv calibration.json \
    original schedule.py calibrate.py preflight.py run.sh assets/weight-sha256.json assets/reference.json \
    assets/original-splits.csv assets/gray-strata.csv
mv distillation-results.tar.gz.part distillation-results.tar.gz
sha256sum distillation-results.tar.gz
