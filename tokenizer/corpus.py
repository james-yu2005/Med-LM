"""Shared paths and corpus token loading for the BPE pipeline."""

from pathlib import Path

import torch

from .bpe_tokenizer import TOKENIZER_VERSION, BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "input.txt"
TOKENIZER_PATH = Path(__file__).resolve().parent / "bpe.json"
CORPUS_CACHE_PATH = Path(__file__).resolve().parent / "corpus_tokens.pt"

BPE_VOCAB_SIZE = 1024
TRAIN_SAMPLE_CHARS = 2_000_000
CORPUS_MAX_CHARS = 4_000_000


def read_corpus(max_chars: int | None = None) -> str:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}. Run: python data/load_dataset.py --format pretrain"
        )
    text = CORPUS_PATH.read_text(encoding="utf-8")
    return text[:max_chars] if max_chars is not None else text


def load_tokenizer() -> BPETokenizer:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}. Run: python -m tokenizer.train"
        )
    return BPETokenizer.load(TOKENIZER_PATH)


def save_corpus_cache(tokens: list[int], max_chars: int = CORPUS_MAX_CHARS) -> None:
    stat = CORPUS_PATH.stat()
    torch.save(
        {
            "tokens": tokens,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "max_chars": max_chars,
            "tokenizer_version": TOKENIZER_VERSION,
        },
        CORPUS_CACHE_PATH,
    )


def load_corpus_tokens(tokenizer: BPETokenizer, max_chars: int = CORPUS_MAX_CHARS) -> list[int]:
    """Load pre-encoded tokens from cache, or encode and cache on miss."""
    corpus_text = read_corpus(max_chars)
    corpus_stat = CORPUS_PATH.stat()

    if CORPUS_CACHE_PATH.exists():
        cache = torch.load(CORPUS_CACHE_PATH, weights_only=False)
        if (
            cache["size"] == corpus_stat.st_size
            and cache["mtime"] == corpus_stat.st_mtime
            and cache.get("max_chars") == max_chars
            and cache.get("tokenizer_version") == TOKENIZER_VERSION
        ):
            return cache["tokens"]

    print("Encoding corpus with BPE (first run or corpus changed)...", flush=True)
    tokens = tokenizer.encode(corpus_text)
    save_corpus_cache(tokens, max_chars)
    return tokens
