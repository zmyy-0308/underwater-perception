#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/subpipe-distillation
python -m venv --system-site-packages .venv
.venv/bin/python -m pip --disable-pip-version-check install --no-index --no-deps staging/wheels/*.whl
tar -xzf distillation-upload.tar.gz
cd distillation
tar -xzf ../code-update.tar.gz
cp ../staging/job-assignment.json job-assignment.json
mkdir -p data/datasets data/manifests
unzip -oq ../staging/subpipe.zip -d data/datasets
source ../.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u - <<'PY'
from pathlib import Path
import hashlib, json
from experiment import ROOT
from tools.prepare_released_segmentation import prepare_pipe
for record in json.loads((ROOT/"assets/weight-sha256.json").read_bytes()):
    assert hashlib.sha256((ROOT/"assets/weights"/record["file"]).read_bytes()).hexdigest()==record["sha256"]
prepare_pipe(ROOT/"data/datasets/subpipe",ROOT/"data/manifests")
PY
python -u original/tools/joint_pipe_cache.py build data/pipe-cache data/manifests/subpipe-fivefold.csv
python -u original/tools/joint_pipe_cache.py verify data/pipe-cache data/manifests/subpipe-fivefold.csv
python -u - <<'PY'
from experiment import audit
import torch
torch.set_num_threads(2)
audit(torch.device("cuda:0"))
PY
export DISTILL_VENV=/root/autodl-tmp/subpipe-distillation/.venv
touch GLOBAL_PRIMARY_BARRIER
printf '12\n' > slots
exec bash distributed.sh 12
