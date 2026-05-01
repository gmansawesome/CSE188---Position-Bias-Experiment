import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

OUTPUT_DIR = Path("data/base_documents")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = Path("data/base_documents_metadata.jsonl")
METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-5.5"

MARKERS = [
    "<<<POSITION_05_PERCENT>>>",
    "<<<POSITION_25_PERCENT>>>",
    "<<<POSITION_50_PERCENT>>>",
    "<<<POSITION_75_PERCENT>>>",
    "<<<POSITION_95_PERCENT>>>",
]

# 10 documents
SUBJECTS = [
    {
        "item_id": "item_001",
        "subject": "maintenance procedures at a remote research facility",
    },
    {
        "item_id": "item_002",
        "subject": "administrative operations at a regional logistics depot",
    },
    {
        "item_id": "item_003",
        "subject": "safety documentation for an underground transit facility",
    },
    {
        "item_id": "item_004",
        "subject": "routine inspection procedures at a coastal monitoring station",
    },
    {
        "item_id": "item_005",
        "subject": "inventory management practices at a municipal storage complex",
    },
    {
        "item_id": "item_006",
        "subject": "training and compliance procedures at a technical support center",
    },
    {
        "item_id": "item_007",
        "subject": "facility access policies at a university research annex",
    },
    {
        "item_id": "item_008",
        "subject": "equipment review procedures at an environmental testing laboratory",
    },
    {
        "item_id": "item_009",
        "subject": "operations planning at a regional emergency coordination office",
    },
    {
        "item_id": "item_010",
        "subject": "document retention practices at a public infrastructure office",
    },
]

LENGTHS = [2000, 5000]


def build_prompt(x_words: int, subject: str) -> str:
    return f"""Generate an approximately {x_words}-word document about {subject}.

Write it in a neutral, dispassionate report style. The document should be coherent but not overly flowery.

Insert the following marker lines at approximately these positions in the document:

<<<POSITION_05_PERCENT>>>
at about 5% of the way through the document

<<<POSITION_25_PERCENT>>>
at about 25% of the way through the document

<<<POSITION_50_PERCENT>>>
at about 50% of the way through the document

<<<POSITION_75_PERCENT>>>
at about 75% of the way through the document

<<<POSITION_95_PERCENT>>>
at about 95% of the way through the document

Rules:
- Do not explain the markers or explain your thought process, only output the document.
- Do not include a title unless it counts as part of the document.
- Each marker should appear ONLY ONCE !!!!
- Each marker must be on its own separate line.
- Do not include bullet points, numbered lists, tables, section headings, or anything like that !!!!
"""


def count_words(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text))


def validate_markers(text: str) -> Dict[str, object]:
    counts = {marker: text.count(marker) for marker in MARKERS}

    missing = [marker for marker, count in counts.items() if count == 0]
    repeated = [marker for marker, count in counts.items() if count > 1]

    valid = not missing and not repeated

    return {
        "valid": valid,
        "counts": counts,
        "missing": missing,
        "repeated": repeated,
    }


def generate_document(prompt: str) -> str:
    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    return response.output_text.strip()


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_file_for(item_id: str, x_words: int) -> Path:
    return OUTPUT_DIR / f"{item_id}_{x_words}_base.txt"


def already_done(item_id: str, x_words: int) -> bool:
    path = output_file_for(item_id, x_words)
    if not path.exists():
        return False

    text = path.read_text(encoding="utf-8")
    return validate_markers(text)["valid"]


def main() -> None:
    for item in SUBJECTS:
        item_id = item["item_id"]
        subject = item["subject"]

        for x_words in LENGTHS:
            out_path = output_file_for(item_id, x_words)

            if already_done(item_id, x_words):
                print(f"Skipping existing valid file: {out_path}")
                continue

            prompt = build_prompt(x_words=x_words, subject=subject)

            print(f"Generating {item_id}, {x_words} words...")
            text = generate_document(prompt)

            marker_report = validate_markers(text)
            actual_words = count_words(text)

            out_path.write_text(text, encoding="utf-8")

            metadata = {
                "item_id": item_id,
                "subject": subject,
                "target_words": x_words,
                "actual_words": actual_words,
                "file_path": str(out_path),
                "marker_valid": marker_report["valid"],
                "marker_counts": marker_report["counts"],
                "missing_markers": marker_report["missing"],
                "repeated_markers": marker_report["repeated"],
                "model": MODEL,
            }

            append_jsonl(METADATA_PATH, metadata)

            if marker_report["valid"]:
                print(f"Saved valid document: {out_path} ({actual_words} words)")
            else:
                print(f"Saved document with marker issue: {out_path}")
                print(json.dumps(marker_report, indent=2))

            time.sleep(1)


if __name__ == "__main__":
    main()