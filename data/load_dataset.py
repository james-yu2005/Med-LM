"""Download medical flashcards and write a training corpus."""

import argparse
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent
PRETRAIN_PATH = DATA_DIR / "input.txt"
INSTRUCT_PATH = DATA_DIR / "instruct.txt"
DATASET_NAME = "medalpaca/medical_meadow_medical_flashcards"

# Fixed labels — training and inference must use the same strings (step 4).
INSTRUCTION_DEFAULT = "Answer this question truthfully"
QUESTION_PREFIX = "Question: "
ANSWER_PREFIX = "Answer: "


def format_qa_example(
    instruction: str,
    question: str,
    answer: str,
) -> str:
    """One instruction-tuning example as plain text."""
    instruction = instruction.strip()
    question = question.strip()
    answer = answer.strip()
    return (
        f"{instruction}\n"
        f"{QUESTION_PREFIX}{question}\n"
        f"{ANSWER_PREFIX}{answer}"
    )


def build_pretrain_corpus(dataset, max_examples: int | None) -> str:
    """Answer-only text stream (original language-modeling format)."""
    rows = dataset[:max_examples] if max_examples else dataset
    return "\n".join(rows["output"])


def build_instruct_corpus(dataset, max_examples: int | None) -> str:
    """Q&A examples separated by blank lines (instruction-tuning format)."""
    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    examples = [
        format_qa_example(
            dataset[i]["instruction"],
            dataset[i]["input"],
            dataset[i]["output"],
        )
        for i in range(n)
    ]
    return "\n\n".join(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and format the medical flashcard dataset.")
    parser.add_argument(
        "--format",
        choices=("pretrain", "instruct"),
        default="instruct",
        help="pretrain: answer-only text for LM. instruct: Q&A pairs for question answering.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50_000,
        help="Max dataset rows to include (default: 50000).",
    )
    args = parser.parse_args()

    print(f"Downloading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")

    if args.format == "pretrain":
        text = build_pretrain_corpus(dataset, args.max_examples)
        output_path = PRETRAIN_PATH
    else:
        text = build_instruct_corpus(dataset, args.max_examples)
        output_path = INSTRUCT_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    example_count = text.count("\n\n") + 1 if args.format == "instruct" else args.max_examples
    print(f"Format:   {args.format}")
    print(f"Examples: {example_count:,}")
    print(f"Saved {len(text):,} characters to {output_path}")

    if args.format == "instruct":
        preview = text.split("\n\n")[0]
        print("\nFirst example preview:\n")
        print(preview)


if __name__ == "__main__":
    main()
