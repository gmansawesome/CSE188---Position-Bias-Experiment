import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


load_dotenv()

client = OpenAI()

DEFAULT_DATASET_PATH = Path("data/dataset.jsonl")
DEFAULT_RESULTS_PATH = Path("data/results.jsonl")
MODEL = "gpt-5.4-nano"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Bad JSON on line {line_num} of {path}: {exc}") from exc


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_keys(results_path: Path) -> set[Tuple[str, str]]:
    """
    Only completed rows count as finished.
    Error rows are not treated as completed, so rerunning can retry them.
    """
    completed: set[Tuple[str, str]] = set()

    if not results_path.exists():
        return completed

    for row in read_jsonl(results_path):
        if row.get("status") == "completed":
            completed.add((row["doc_id"], row["prompting_method"]))

    return completed


def normalize_answer_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text)
    text = text.rstrip(".。,:;!?)\"]}'")
    return text


def is_correct(model_output: str, correct_answer: str) -> bool:
    """
    Simple exact/string-containment scoring.

    This is appropriate here because your answers are artificial codes like QK-4829.
    """
    output_norm = normalize_answer_text(model_output)
    answer_norm = normalize_answer_text(correct_answer)
    return answer_norm in output_norm


def count_words(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text))


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def chunk_words(text: str, chunk_size: int = 200, overlap: int = 50) -> List[Dict[str, Any]]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks: List[Dict[str, Any]] = []

    start = 0
    chunk_id = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "start_word": start,
                "end_word": end,
            }
        )

        chunk_id += 1
        start += step

    return chunks


def retrieve_bm25(
    document_text: str,
    question: str,
    top_k: int = 3,
    chunk_size: int = 200,
    overlap: int = 50,
) -> Tuple[List[Dict[str, Any]], str]:
    if BM25Okapi is None:
        raise ImportError(
            "rank-bm25 is not installed. Run: pip install rank-bm25"
        )

    chunks = chunk_words(document_text, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        return [], ""

    tokenized_chunks = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    query_tokens = tokenize(question)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    top_chunks = [chunks[i] for i in ranked_indices[:top_k]]

    retrieved_context = "\n\n".join(
        f"[Chunk {chunk['chunk_id']}]\n{chunk['text']}"
        for chunk in top_chunks
    )

    return top_chunks, retrieved_context


def answer_was_retrieved(
    retrieved_context: str,
    correct_answer: str,
    answer_sentence: str,
) -> bool:
    retrieved_lower = retrieved_context.lower()
    return (
        correct_answer.lower() in retrieved_lower
        or answer_sentence.lower() in retrieved_lower
    )


def build_full_context_prompt(document_text: str, question: str) -> str:
    return f"""You will be given a document and a question.

Use only the information in the document.
Answer with the exact answer only.
If the answer is not present, write NOT FOUND.

Document:
{document_text}

Question:
{question}

Answer:"""


def build_bm25_prompt(retrieved_context: str, question: str) -> str:
    return f"""You will be given retrieved passages from a longer document and a question.

Use only the retrieved passages.
Answer with the exact answer only.
If the answer is not present, write NOT FOUND.

Retrieved passages:
{retrieved_context}

Question:
{question}

Answer:"""


def call_model(
    prompt: str,
    model: str,
    max_output_tokens: int,
    max_retries: int = 3,
    retry_sleep: float = 3.0,
) -> str:
    """
    Calls the OpenAI Responses API and returns output_text.

    max_output_tokens should stay small because the expected answer is only a code.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            return response.output_text.strip()

        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break

            sleep_for = retry_sleep * attempt
            print(
                f"API error on attempt {attempt}/{max_retries}: {exc}. "
                f"Retrying in {sleep_for:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_for)

    raise RuntimeError(f"API call failed after {max_retries} attempts: {last_error}")


def validate_dataset_row(row: Dict[str, Any]) -> None:
    required = [
        "doc_id",
        "item_id",
        "word_count_target",
        "filler_condition",
        "answer_position",
        "question",
        "correct_answer",
        "answer_sentence",
        "document_text",
    ]

    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"Dataset row {row.get('doc_id', '<unknown>')} is missing keys: {missing}")

    if row["correct_answer"] not in row["document_text"]:
        raise ValueError(
            f"{row['doc_id']} does not contain correct_answer {row['correct_answer']!r}"
        )

    count = row["document_text"].count(row["correct_answer"])
    if count != 1:
        raise ValueError(
            f"{row['doc_id']} contains correct_answer {row['correct_answer']!r} {count} times; expected exactly 1."
        )

    leftover_markers = re.findall(r"<<<POSITION_\d+_PERCENT>>>", row["document_text"])
    if leftover_markers:
        raise ValueError(
            f"{row['doc_id']} still contains marker(s): {sorted(set(leftover_markers))}"
        )


def make_base_result(row: Dict[str, Any], method: str, model: str) -> Dict[str, Any]:
    return {
        "doc_id": row["doc_id"],
        "item_id": row["item_id"],
        "word_count_target": row["word_count_target"],
        "filler_condition": row["filler_condition"],
        "answer_position": row["answer_position"],
        "prompting_method": method,
        "model": model,
        "question": row["question"],
        "correct_answer": row["correct_answer"],
        "answer_sentence": row["answer_sentence"],
        "document_word_count": count_words(row["document_text"]),
        "created_at": utc_now_iso(),
    }


def run_full_context(
    row: Dict[str, Any],
    model: str,
    max_output_tokens: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    prompt = build_full_context_prompt(row["document_text"], row["question"])

    result = make_base_result(row, method="full_context", model=model)
    result["prompt_word_count"] = count_words(prompt)

    if dry_run:
        result.update(
            {
                "status": "dry_run",
                "model_output": "",
                "is_correct": None,
            }
        )
        return result

    output = call_model(
        prompt=prompt,
        model=model,
        max_output_tokens=max_output_tokens,
    )

    result.update(
        {
            "status": "completed",
            "model_output": output,
            "is_correct": is_correct(output, row["correct_answer"]),
        }
    )
    return result


def run_bm25(
    row: Dict[str, Any],
    model: str,
    max_output_tokens: int,
    top_k: int,
    chunk_size: int,
    overlap: int,
    store_retrieved_context: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    top_chunks, retrieved_context = retrieve_bm25(
        document_text=row["document_text"],
        question=row["question"],
        top_k=top_k,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    prompt = build_bm25_prompt(retrieved_context, row["question"])

    retrieved = answer_was_retrieved(
        retrieved_context=retrieved_context,
        correct_answer=row["correct_answer"],
        answer_sentence=row["answer_sentence"],
    )

    result = make_base_result(row, method="bm25", model=model)
    result.update(
        {
            "bm25_top_k": top_k,
            "bm25_chunk_size": chunk_size,
            "bm25_overlap": overlap,
            "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in top_chunks],
            "answer_chunk_retrieved": retrieved,
            "retrieved_context_word_count": count_words(retrieved_context),
            "prompt_word_count": count_words(prompt),
        }
    )

    if store_retrieved_context:
        result["retrieved_context"] = retrieved_context

    if dry_run:
        result.update(
            {
                "status": "dry_run",
                "model_output": "",
                "is_correct": None,
            }
        )
        return result

    output = call_model(
        prompt=prompt,
        model=model,
        max_output_tokens=max_output_tokens,
    )

    result.update(
        {
            "status": "completed",
            "model_output": output,
            "is_correct": is_correct(output, row["correct_answer"]),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-context and/or BM25 QA experiment over data/dataset.jsonl."
    )

    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=30)

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["full_context", "bm25"],
        default=["full_context", "bm25"],
        help="Which prompting methods to run.",
    )

    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset rows for a pilot.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset rows before running.")
    parser.add_argument("--seed", default="experiment-seed", help="Seed used when --shuffle is enabled.")

    parser.add_argument("--bm25-top-k", type=int, default=3)
    parser.add_argument("--bm25-chunk-size", type=int, default=200)
    parser.add_argument("--bm25-overlap", type=int, default=50)
    parser.add_argument(
        "--store-retrieved-context",
        action="store_true",
        help="Store retrieved BM25 text in results.jsonl. Useful for debugging but increases file size.",
    )

    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each completed API call.")
    parser.add_argument("--dry-run", action="store_true", help="Build result rows without calling the API.")
    parser.add_argument("--no-validate", action="store_true", help="Skip dataset sanity checks.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Missing dataset file: {args.dataset}")

    rows = list(read_jsonl(args.dataset))

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    if args.limit is not None:
        rows = rows[: args.limit]

    completed = load_completed_keys(args.results)

    print(f"Dataset rows selected: {len(rows)}")
    print(f"Methods: {args.methods}")
    print(f"Model: {args.model}")
    print(f"Existing completed result rows: {len(completed)}")
    print(f"Writing results to: {args.results}")

    attempted = 0
    skipped = 0
    completed_now = 0
    errors = 0

    for index, row in enumerate(rows, start=1):
        if not args.no_validate:
            validate_dataset_row(row)

        for method in args.methods:
            key = (row["doc_id"], method)

            if key in completed:
                skipped += 1
                print(f"[{index}/{len(rows)}] SKIP {row['doc_id']} / {method}")
                continue

            attempted += 1
            print(f"[{index}/{len(rows)}] RUN  {row['doc_id']} / {method}")

            try:
                if method == "full_context":
                    result = run_full_context(
                        row=row,
                        model=args.model,
                        max_output_tokens=args.max_output_tokens,
                        dry_run=args.dry_run,
                    )

                elif method == "bm25":
                    result = run_bm25(
                        row=row,
                        model=args.model,
                        max_output_tokens=args.max_output_tokens,
                        top_k=args.bm25_top_k,
                        chunk_size=args.bm25_chunk_size,
                        overlap=args.bm25_overlap,
                        store_retrieved_context=args.store_retrieved_context,
                        dry_run=args.dry_run,
                    )

                else:
                    raise ValueError(f"Unknown method: {method}")

                append_jsonl(args.results, result)

                if result["status"] == "completed":
                    completed_now += 1
                    completed.add(key)

                print(
                    f"       output={result.get('model_output', '')!r} "
                    f"correct={result.get('is_correct')}"
                )

                if args.sleep > 0:
                    time.sleep(args.sleep)

            except Exception as exc:
                errors += 1
                error_row = make_base_result(row, method=method, model=args.model)
                error_row.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                append_jsonl(args.results, error_row)
                print(f"       ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)

    print("\nDone.")
    print(f"Attempted this run: {attempted}")
    print(f"Completed this run: {completed_now}")
    print(f"Skipped existing: {skipped}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
