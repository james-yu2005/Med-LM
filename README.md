# MedLM — Transformers from Scratch

A from-scratch language modeling project in PyTorch, inspired by [Karpathy's "Let's build GPT" series](https://www.youtube.com/watch?v=kCc8FmEb1nY). Trains on medical flashcard text with a progression from simple bigram models to a full GPT with a byte-level BPE tokenizer.

## What this project does

1. **Bigram model** — predicts the next character from only the previous one
2. **GPT + BPE** — a ~10M-parameter Transformer trained on subword tokens (byte-level BPE, GPT-2 style)
3. **Custom tokenizer** — BPE trained from scratch on UTF-8 bytes (no HuggingFace tokenizer)
4. **GPU training pipeline** — train locally or on RunPod / any cloud GPU, with checkpoint saving and a downloadable model bundle

## Project structure

```
transformers/
├── data/
│   ├── load_dataset.py    # download & format medical flashcards
│   ├── input.txt          # pretrain corpus (generated)
│   └── instruct.txt       # Q&A corpus for future instruction tuning (generated)
├── tokenizer/
│   ├── bpe_tokenizer.py   # byte-level BPE encode/decode/train
│   ├── corpus.py          # paths, corpus loading, token cache
│   ├── train.py           # train tokenizer + encode corpus
│   ├── bpe.json           # learned merges (generated)
│   └── corpus_tokens.pt   # pre-encoded training data (generated)
├── checkpoints/
│   ├── gpt_bpe.pt         # latest eval snapshot
│   ├── gpt_bpe_best.pt    # best validation loss (use for generation)
│   ├── gpt_bpe_final.pt   # final training step
│   └── model_bundle.zip   # weights + tokenizer for download
├── scripts/
│   └── runpod_train.sh    # one-shot training script for cloud GPUs
├── bigram.py              # step 1: character bigram baseline
├── gpt_bpe.py             # step 2: GPT on BPE tokens (train / generate)
├── export_bundle.py       # zip checkpoint + tokenizer for backup
├── test_encode.py         # interactive tokenizer tester
└── requirements.txt
```

## Requirements

- Python 3.10+
- PyTorch (included on RunPod PyTorch templates; install locally with [pytorch.org](https://pytorch.org))
- `datasets` (for downloading the medical corpus)

```powershell
pip install -r requirements.txt
```

On RunPod, use a **PyTorch** template — CUDA PyTorch is pre-installed. `requirements.txt` only adds `datasets` and pins `numpy<2`.

## Quick start (local CPU or GPU)

Run everything from the project root (`transformers/`).

### 1. Get training data

```powershell
python data/load_dataset.py --format pretrain
```

This downloads medical flashcards and writes answer-only text to `data/input.txt` (~50k examples).

For Q&A-formatted data (future instruction tuning):

```powershell
python data/load_dataset.py --format instruct
```

Writes `data/instruct.txt`. The current GPT trainer uses `input.txt` (pretrain format).

### 2. Train tokenizer and encode corpus

```powershell
python -m tokenizer.train
```

This will:
- Train byte-level BPE on the first **2M characters** (768 merges → 1024 vocab)
- Encode the first **4M characters** for GPT training
- Save `tokenizer/bpe.json` and `tokenizer/corpus_tokens.pt`

To re-encode only (tokenizer already exists):

```powershell
python -m tokenizer.train --cache-only
```

### 3. Train the GPT

```powershell
python gpt_bpe.py
```

- Auto-detects CUDA GPU; falls back to CPU if unavailable
- Saves checkpoints every 500 steps to `checkpoints/`
- Keeps the best validation checkpoint as `checkpoints/gpt_bpe_best.pt`
- Exports `checkpoints/model_bundle.zip` when training finishes

### 4. Generate text from a trained checkpoint

```powershell
python gpt_bpe.py --generate --prompt "Diabetes is"
python gpt_bpe.py --generate --checkpoint checkpoints/gpt_bpe_best.pt --prompt "Heart failure is"
```

Generation only needs `checkpoints/gpt_bpe_best.pt` and `tokenizer/bpe.json`.

### 5. Resume training

```powershell
python gpt_bpe.py --resume
```

## Train on RunPod (or any cloud GPU)

### Recommended GPU

| GPU | Notes |
|-----|-------|
| **RTX 4090 / 3090** | Best value (~15–25 min training) |
| **RTX A4000 / T4** | Cheaper, slower |

Any GPU with **≥16 GB VRAM** is more than enough for this ~10M-parameter model.

### Setup

1. Push this repo to GitHub
2. Deploy a RunPod pod with a **PyTorch** template (Community Cloud is fine)
3. Add your SSH public key in RunPod settings (optional — Web Terminal works without SSH)
4. Open **Connect → Web Terminal** (or SSH in)

### One-command training

```bash
git clone https://github.com/james-yu2005/mini-medical-gpt.git
cd mini-medical-gpt
bash scripts/runpod_train.sh
```

Or step by step:

```bash
git clone https://github.com/james-yu2005/mini-medical-gpt.git
cd mini-medical-gpt
pip install -r requirements.txt
python data/load_dataset.py --format pretrain
python -m tokenizer.train
python gpt_bpe.py
python export_bundle.py
```

### Download your model before stopping the pod

RunPod container disk is **ephemeral** — files are deleted when the pod terminates.

Download **`checkpoints/model_bundle.zip`** (contains `gpt_bpe_best.pt` + `bpe.json`):

- **Jupyter Terminal:** `curl -F "file=@checkpoints/model_bundle.zip" https://0x0.st` → open the URL on your PC
- **Jupyter file browser:** download `model_bundle.zip` from `checkpoints/` (large folders may be slow in the UI)
- **SCP from your PC:**

```powershell
scp -P PORT -i $env:USERPROFILE\.ssh\id_ed25519 root@POD_IP:/workspace/mini-medical-gpt/checkpoints/model_bundle.zip .
```

### Use the model locally

```powershell
cd C:\Users\james\Desktop\transformers
Expand-Archive -Path .\model_bundle.zip -DestinationPath . -Force
python gpt_bpe.py --generate --prompt "Diabetes is"
```

You need these two files in the project:

```
checkpoints/gpt_bpe_best.pt
tokenizer/bpe.json
```

## Model details

### Bigram (`bigram.py`)

| Setting | Value |
|---------|-------|
| Token level | Character |
| Context | 8 chars |
| Parameters | ~4K |

### GPT (`gpt_bpe.py`)

| Setting | Value |
|---------|-------|
| Token level | Byte BPE (1024 vocab) |
| Context (`block_size`) | 256 tokens |
| `n_embd` | 384 |
| Layers / heads | 6 / 6 |
| Parameters | ~10M |
| Training steps | 5000 |

### BPE tokenizer

| Setting | Value |
|---------|-------|
| Type | Byte-level BPE (raw UTF-8) |
| Base vocab | 256 bytes |
| Merges | 768 |
| Train sample | First 2M chars of corpus |
| LM corpus | First 4M chars of corpus |

Limits are in `tokenizer/corpus.py` (`TRAIN_SAMPLE_CHARS`, `CORPUS_MAX_CHARS`).

## How BPE fits in

```
text  →  UTF-8 bytes [0–255]  →  apply merges  →  token IDs  →  GPT
IDs   →  lookup bytes per ID  →  UTF-8 decode   →  text
```

- `bpe.json` — merge rules (required for encode/decode)
- `corpus_tokens.pt` — pre-encoded integer sequence for fast training startup

## Checkpoints

| File | Purpose |
|------|---------|
| `checkpoints/gpt_bpe.pt` | Latest snapshot (every 500 steps) |
| `checkpoints/gpt_bpe_best.pt` | Best validation loss — **use this for generation** |
| `checkpoints/gpt_bpe_final.pt` | Weights from the final step |
| `checkpoints/model_bundle.zip` | Portable download: best weights + `bpe.json` |

Export the bundle manually anytime:

```powershell
python export_bundle.py
```

## Learning progression

```
bigram.py          gpt_bpe.py
    │                  │
    │                  ├── tokenizer/bpe_tokenizer.py
    │                  ├── self-attention (6 heads)
    │                  ├── 6 transformer blocks
    │                  └── byte BPE subwords
    │
    └── 1-char memory, embedding lookup only
```

## Generated files (gitignored)

These are created locally and not committed:

- `data/input.txt`
- `data/instruct.txt`
- `tokenizer/bpe.json`
- `tokenizer/corpus_tokens.pt`
- `checkpoints/`
- `gpt_venv/`
