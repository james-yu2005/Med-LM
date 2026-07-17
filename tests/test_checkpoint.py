"""Checkpoint persistence, resume edge cases, and bundle defaults."""

import zipfile

import torch
import torch.nn as nn

import gpt_bpe
from export_bundle import export_bundle, resolve_default_checkpoint
from gpt_bpe import load_checkpoint, read_checkpoint_meta, save_checkpoint, train


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Linear(4, 4)


def test_save_load_roundtrip_preserves_best_val_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(gpt_bpe, "device", "cpu")
    path = tmp_path / "ckpt.pt"
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = {"train": 1.25, "val": 0.75}

    save_checkpoint(path, step=42, model=model, optimizer=optimizer, vocab_size=32, losses=losses, best_val_loss=0.75)

    model2 = TinyModel()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    step, best_val = load_checkpoint(path, model2, vocab_size=32, optimizer=optimizer2)

    assert step == 42
    assert best_val == 0.75
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)

    meta = read_checkpoint_meta(path)
    assert meta["step"] == 42
    assert meta["best_val_loss"] == 0.75
    assert meta["losses"] == losses


def test_read_checkpoint_meta_falls_back_to_losses_val(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "step": 7,
            "model_state_dict": TinyModel().state_dict(),
            "config": {},
            "losses": {"train": 2.0, "val": 1.5},
        },
        path,
    )
    meta = read_checkpoint_meta(path)
    assert meta["best_val_loss"] == 1.5


def test_resume_restores_best_val_and_skips_finished_run(tmp_path, monkeypatch):
    monkeypatch.setattr(gpt_bpe, "device", "cpu")
    monkeypatch.setattr(gpt_bpe, "max_iters", 2)
    monkeypatch.setattr(gpt_bpe, "eval_interval", 1)
    monkeypatch.setattr(gpt_bpe, "eval_iters", 1)
    monkeypatch.setattr(gpt_bpe, "batch_size", 2)
    monkeypatch.setattr(gpt_bpe, "block_size", 8)
    monkeypatch.setattr(gpt_bpe, "n_embd", 16)
    monkeypatch.setattr(gpt_bpe, "n_head", 2)
    monkeypatch.setattr(gpt_bpe, "n_layer", 1)
    monkeypatch.setattr(gpt_bpe, "dropout", 0.0)
    monkeypatch.setattr(gpt_bpe, "BEST_CHECKPOINT", tmp_path / "gpt_bpe_best.pt")
    monkeypatch.setattr(gpt_bpe, "FINAL_CHECKPOINT", tmp_path / "gpt_bpe_final.pt")

    vocab_size = 32
    model = gpt_bpe.GPTLanguageModel(vocab_size).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = torch.randint(0, vocab_size, (64,), dtype=torch.long)
    train_data, val_data = data[:48], data[48:]
    checkpoint_path = tmp_path / "gpt_bpe.pt"

    best_path = train(
        model, optimizer, train_data, val_data, checkpoint_path, vocab_size, resume=False
    )
    assert best_path.exists()
    assert checkpoint_path.exists()

    meta_before = read_checkpoint_meta(gpt_bpe.BEST_CHECKPOINT)
    assert meta_before["best_val_loss"] is not None

    # Finished run: next step would be max_iters; must not crash on unbound losses.
    best_again = train(
        model, optimizer, train_data, val_data, checkpoint_path, vocab_size, resume=True
    )
    assert best_again == gpt_bpe.BEST_CHECKPOINT
    meta_after = read_checkpoint_meta(gpt_bpe.BEST_CHECKPOINT)
    assert meta_after["best_val_loss"] == meta_before["best_val_loss"]


def test_resume_keeps_historical_best_when_val_regresses(tmp_path, monkeypatch):
    monkeypatch.setattr(gpt_bpe, "device", "cpu")
    monkeypatch.setattr(gpt_bpe, "max_iters", 3)
    monkeypatch.setattr(gpt_bpe, "eval_interval", 1)
    monkeypatch.setattr(gpt_bpe, "eval_iters", 1)
    monkeypatch.setattr(gpt_bpe, "batch_size", 2)
    monkeypatch.setattr(gpt_bpe, "block_size", 8)
    monkeypatch.setattr(gpt_bpe, "n_embd", 16)
    monkeypatch.setattr(gpt_bpe, "n_head", 2)
    monkeypatch.setattr(gpt_bpe, "n_layer", 1)
    monkeypatch.setattr(gpt_bpe, "dropout", 0.0)
    monkeypatch.setattr(gpt_bpe, "BEST_CHECKPOINT", tmp_path / "gpt_bpe_best.pt")
    monkeypatch.setattr(gpt_bpe, "FINAL_CHECKPOINT", tmp_path / "gpt_bpe_final.pt")

    vocab_size = 32
    model = gpt_bpe.GPTLanguageModel(vocab_size).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data = torch.randint(0, vocab_size, (64,), dtype=torch.long)
    train_data, val_data = data[:48], data[48:]
    checkpoint_path = tmp_path / "gpt_bpe.pt"

    # Seed a "best" checkpoint that is better than anything short training will beat.
    save_checkpoint(
        gpt_bpe.BEST_CHECKPOINT,
        step=0,
        model=model,
        optimizer=optimizer,
        vocab_size=vocab_size,
        losses={"train": 0.01, "val": 0.01},
        best_val_loss=0.01,
    )
    save_checkpoint(
        checkpoint_path,
        step=0,
        model=model,
        optimizer=optimizer,
        vocab_size=vocab_size,
        losses={"train": 0.01, "val": 0.01},
        best_val_loss=0.01,
    )

    monkeypatch.setattr(gpt_bpe, "max_iters", 2)
    train(model, optimizer, train_data, val_data, checkpoint_path, vocab_size, resume=True)

    best_meta = read_checkpoint_meta(gpt_bpe.BEST_CHECKPOINT)
    latest_meta = read_checkpoint_meta(checkpoint_path)
    assert best_meta["best_val_loss"] == 0.01
    assert latest_meta["best_val_loss"] == 0.01


def test_resolve_default_checkpoint_prefers_best(tmp_path, monkeypatch):
    best = tmp_path / "gpt_bpe_best.pt"
    latest = tmp_path / "gpt_bpe.pt"
    best.write_bytes(b"best")
    latest.write_bytes(b"latest")
    monkeypatch.setattr("export_bundle.BEST_CHECKPOINT", best)
    monkeypatch.setattr("export_bundle.DEFAULT_CHECKPOINT", latest)
    assert resolve_default_checkpoint() == best


def test_resolve_default_checkpoint_falls_back_to_latest(tmp_path, monkeypatch):
    best = tmp_path / "gpt_bpe_best.pt"
    latest = tmp_path / "gpt_bpe.pt"
    latest.write_bytes(b"latest")
    monkeypatch.setattr("export_bundle.BEST_CHECKPOINT", best)
    monkeypatch.setattr("export_bundle.DEFAULT_CHECKPOINT", latest)
    assert resolve_default_checkpoint() == latest


def test_export_bundle_packs_requested_checkpoint(tmp_path):
    ckpt = tmp_path / "gpt_bpe_best.pt"
    tok = tmp_path / "bpe.json"
    out = tmp_path / "model_bundle.zip"
    ckpt.write_bytes(b"weights")
    tok.write_text("{}", encoding="utf-8")

    export_bundle(ckpt, tok, out)

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert names == {"checkpoints/gpt_bpe_best.pt", "tokenizer/bpe.json"}
