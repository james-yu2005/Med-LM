#!/usr/bin/env bash
# Full training pipeline for RunPod (PyTorch template already includes torch+CUDA).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== GPU check ==="
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "=== Install deps ==="
pip install -q -r requirements.txt

echo "=== Download data ==="
python data/load_dataset.py --format pretrain

echo "=== Train tokenizer ==="
python -m tokenizer.train

echo "=== Train GPT ==="
python gpt_bpe.py "$@"

echo "=== Export bundle for download ==="
python export_bundle.py

echo "Done. Download checkpoints/model_bundle.zip before stopping the pod."
