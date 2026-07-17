"""GPT trained on byte-level BPE tokens. Run: python -m tokenizer.train  then  python gpt_bpe.py"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer.corpus import CORPUS_MAX_CHARS, load_corpus_tokens, load_tokenizer

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "gpt_bpe.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "gpt_bpe_best.pt"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "gpt_bpe_final.pt"

def get_device() -> str:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("CUDA not available - training on CPU (slow). Use a RunPod PyTorch template for GPU.")
    return "cpu"

device = get_device()

torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed_all(1337)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=block_size, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        # Interleave so each adjacent (even, odd) pair shares one frequency.
        emb = torch.stack((freqs, freqs), dim=-1).flatten(-2)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x, offset=0):
        T = x.size(1)
        cos = self.cos[offset : offset + T].unsqueeze(0)
        sin = self.sin[offset : offset + T].unsqueeze(0)
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        rot = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)
        return x * cos + rot * sin

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope):
        B, T, C = x.shape
        q = rope(self.query(x))
        k = rope(self.key(x))
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope):
        out = torch.cat([h(x, rope) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, rope):
        x = x + self.sa(self.ln1(x), rope)
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.rope = RotaryEmbedding(n_embd // n_head)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        x = self.token_embedding_table(idx) # (B,T,C)
        for block in self.blocks:
            x = block(x, self.rope)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def model_config(vocab_size: int) -> dict:
    return {
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
    }

def save_checkpoint(
    path: Path,
    step: int,
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer | None,
    vocab_size: int,
    losses: dict | None = None,
    best_val_loss: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": model_config(vocab_size),
        "losses": losses,
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    print(f"Saved checkpoint: {path}")

def read_checkpoint_meta(path: Path) -> dict:
    """Load checkpoint metadata without applying weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    best_val_loss = checkpoint.get("best_val_loss")
    losses = checkpoint.get("losses")
    if best_val_loss is None and isinstance(losses, dict):
        best_val_loss = losses.get("val")
    return {
        "step": checkpoint.get("step", checkpoint.get("iter", 0)),
        "best_val_loss": best_val_loss,
        "losses": losses,
        "config": checkpoint.get("config", {}),
    }

def load_checkpoint(
    path: Path,
    model: GPTLanguageModel,
    vocab_size: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, float | None]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config", {})
    for key, value in model_config(vocab_size).items():
        if saved_config.get(key) != value:
            print(f"Warning: checkpoint {key}={saved_config.get(key)!r} differs from current {value!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint.get("step", checkpoint.get("iter", 0))
    best_val_loss = checkpoint.get("best_val_loss")
    print(f"Loaded checkpoint from step {step}: {path}")
    return step, best_val_loss

def generate_sample(
    model: GPTLanguageModel,
    encode,
    decode,
    prompt: str = "Diabetes is",
    max_new_tokens: int = 500,
) -> str:
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(context, max_new_tokens=max_new_tokens)
    return decode(output[0].tolist())

def train(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    checkpoint_path: Path,
    vocab_size: int,
    resume: bool = False,
) -> Path:
    """Train the model. Returns path to the best checkpoint (lowest val loss)."""
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters, device=device)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    start_step = 0
    best_val = float("inf")
    best_path = checkpoint_path
    losses = None

    if resume and checkpoint_path.exists():
        start_step, restored_best = load_checkpoint(checkpoint_path, model, vocab_size, optimizer)
        start_step += 1
        if restored_best is not None:
            best_val = restored_best
        if BEST_CHECKPOINT.exists():
            best_path = BEST_CHECKPOINT
            meta_best = read_checkpoint_meta(BEST_CHECKPOINT).get("best_val_loss")
            if meta_best is not None:
                best_val = meta_best

    if start_step >= max_iters:
        print(
            f"Checkpoint already finished training "
            f"(next step {start_step} >= max_iters {max_iters}). Skipping train loop."
        )
        return best_path if BEST_CHECKPOINT.exists() else checkpoint_path

    for step in range(start_step, max_iters):

        # every once in a while evaluate the loss on train and val sets
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(
                    BEST_CHECKPOINT, step, model, optimizer, vocab_size, losses, best_val
                )
                best_path = BEST_CHECKPOINT
                print(f"  new best val loss {best_val:.4f} -> {BEST_CHECKPOINT}")
            save_checkpoint(
                checkpoint_path, step, model, optimizer, vocab_size, losses, best_val
            )

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_checkpoint(
        FINAL_CHECKPOINT, max_iters - 1, model, optimizer, vocab_size, losses, best_val
    )
    print(f"Training complete. Best weights: {best_path}")
    return best_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run a byte-level BPE GPT.")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Load a checkpoint and generate text (skip training).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: checkpoints/gpt_bpe_best.pt if present, else gpt_bpe.pt).",
    )
    parser.add_argument(
        "--prompt",
        default="Diabetes is",
        help="Prompt used with --generate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Tokens to generate with --generate.",
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        if args.generate and BEST_CHECKPOINT.exists():
            args.checkpoint = BEST_CHECKPOINT
        elif args.generate and FINAL_CHECKPOINT.exists():
            args.checkpoint = FINAL_CHECKPOINT
        else:
            args.checkpoint = DEFAULT_CHECKPOINT

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.vocab_size
    encode = tokenizer.encode
    decode = tokenizer.decode

    model = GPTLanguageModel(vocab_size).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

    if args.generate:
        if not args.checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}. Train first or download your RunPod checkpoint."
            )
        load_checkpoint(args.checkpoint, model, vocab_size)
        model.eval()
        print(generate_sample(model, encode, decode, args.prompt, args.max_new_tokens))
        return

    tokens = load_corpus_tokens(tokenizer, CORPUS_MAX_CHARS)
    print(f"Corpus: {CORPUS_MAX_CHARS:,} chars -> {len(tokens):,} BPE tokens")

    data = torch.tensor(tokens, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if args.resume and not args.checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint to resume: {args.checkpoint}")

    best_path = train(model, optimizer, train_data, val_data, args.checkpoint, vocab_size, resume=args.resume)

    from export_bundle import export_bundle
    from tokenizer.corpus import TOKENIZER_PATH

    bundle_path = CHECKPOINT_DIR / "model_bundle.zip"
    export_bundle(best_path, TOKENIZER_PATH, bundle_path)

    load_checkpoint(best_path, model, vocab_size)
    model.eval()
    print("\nSample generation:")
    print(generate_sample(model, encode, decode))
    print(f"\nTo test later (local or new pod):\n  python gpt_bpe.py --generate --checkpoint {best_path}")


if __name__ == "__main__":
    main()
