#!/usr/bin/env bash
# One-time setup on a RunPod GPU pod (PyTorch template, volume mounted at /workspace).
#   cd /workspace && git clone https://github.com/Drwae/TinyFeeder.git && cd TinyFeeder
#   bash scripts/runpod_setup.sh
# Installs the package, runs the tests, records the environment, downloads AV-FFIA
# to /workspace/AV-FFIA and prints the dataset inspection (send that output back
# before converting).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs logs
apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 wget tmux > /dev/null
pip install -q -e ".[paper,prep,dev]"
pytest -q
{ nvidia-smi; python -c "import torch;print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"; } \
  | tee logs/environment.txt
python scripts/download_avffia.py --out /workspace/AV-FFIA 2>&1 | tee logs/download.log
VID=$(find /workspace/AV-FFIA -maxdepth 3 -type d -iname '*video*' | head -1)
AUD=$(find /workspace/AV-FFIA -maxdepth 3 -type d -iname '*audio*' | head -1)
echo "video root: $VID"; echo "audio root: $AUD"
python scripts/prepare_avffia.py --video-root "$VID" --audio-root "$AUD" --inspect 2>&1 | tee logs/inspect.log
echo; echo "== setup finished. Send logs/inspect.log (shown above) before converting. =="
