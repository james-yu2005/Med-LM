"""Zip model weights + tokenizer for download from RunPod or local backup."""

import argparse
import zipfile
from pathlib import Path

from gpt_bpe import BEST_CHECKPOINT, DEFAULT_CHECKPOINT
from tokenizer.corpus import TOKENIZER_PATH

BUNDLE_PATH = Path(__file__).resolve().parent / "checkpoints" / "model_bundle.zip"


def resolve_default_checkpoint() -> Path:
    """Prefer the best-val checkpoint so bundles stay generation-ready."""
    if BEST_CHECKPOINT.exists():
        return BEST_CHECKPOINT
    return DEFAULT_CHECKPOINT


def export_bundle(
    checkpoint: Path,
    tokenizer_path: Path,
    output: Path,
) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. Run: python -m tokenizer.train"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(checkpoint, f"checkpoints/{checkpoint.name}")
        zf.write(tokenizer_path, f"tokenizer/{tokenizer_path.name}")

    size_mb = output.stat().st_size / 1e6
    print(f"Exported {output} ({size_mb:.1f} MB)")
    print("To use locally: unzip, then run:")
    print(f"  python gpt_bpe.py --generate --checkpoint checkpoints/{checkpoint.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export checkpoint + tokenizer as a zip bundle.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file to include (default: gpt_bpe_best.pt if present, else gpt_bpe.pt).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=TOKENIZER_PATH,
        help="Tokenizer JSON to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BUNDLE_PATH,
        help="Output zip path.",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint if args.checkpoint is not None else resolve_default_checkpoint()
    export_bundle(checkpoint, args.tokenizer, args.output)


if __name__ == "__main__":
    main()
