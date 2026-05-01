import json
import random
import re
from pathlib import Path

BASE_DOCS_DIR = Path("data/base_documents")
OUTPUT_PATH = Path("data/dataset.jsonl")

DATASET_RANDOM_SEED = 188

MARKERS = {
    "beginning": "<<<POSITION_05_PERCENT>>>",
    "early_middle": "<<<POSITION_25_PERCENT>>>",
    "middle": "<<<POSITION_50_PERCENT>>>",
    "late_middle": "<<<POSITION_75_PERCENT>>>",
    "end": "<<<POSITION_95_PERCENT>>>",
}

ALL_MARKERS = list(MARKERS.values())

NEUTRAL_CONTROL_SENTENCES = [
    "The review team noted that routine documentation procedures remained unchanged during this reporting period.",
    "The section emphasized that administrative records should be maintained according to ordinary retention practices.",
    "The report observed that scheduled inspections continued without unusual deviations from standard procedure.",
    "The committee stated that ordinary staff coordination practices were sufficient for the reviewed activities.",
    "The documentation noted that general operational summaries were archived according to existing policy.",
]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_distractor_sentence(distractor):
    return (
        f"The {distractor['label']} was recorded as "
        f"{distractor['answer']} in the auxiliary documentation."
    )


def validate_base_document(text, file_path):
    for marker in ALL_MARKERS:
        count = text.count(marker)
        if count != 1:
            raise ValueError(
                f"{file_path} has marker problem: {marker} appears {count} times."
            )


def build_document(
    base_text,
    answer_sentence,
    selected_marker,
    filler_condition,
    distractors,
    rng,
    fact=None,
):
    text = base_text

    # Insert the true answer at the selected position.
    text = text.replace(selected_marker, answer_sentence)

    # Fill the other four marker locations.
    remaining_markers = [m for m in ALL_MARKERS if m != selected_marker]

    if filler_condition == "generic":
        for i, marker in enumerate(remaining_markers):
            neutral_sentence = NEUTRAL_CONTROL_SENTENCES[i % len(NEUTRAL_CONTROL_SENTENCES)]
            text = text.replace(marker, neutral_sentence)

    elif filler_condition == "distractor":
        if fact is None:
            raise ValueError("fact must be provided when filler_condition is 'distractor'.")

        if len(distractors) < len(remaining_markers):
            raise ValueError(
                f"Need at least {len(remaining_markers)} distractors, "
                f"but only found {len(distractors)}."
            )

        selected_distractors = select_distractors_for_fact(
            fact=fact,
            all_distractors=distractors,
            k=len(remaining_markers),
            rng=rng,
        )

        for marker, distractor in zip(remaining_markers, selected_distractors):
            distractor_sentence = make_distractor_sentence(distractor)
            text = text.replace(marker, distractor_sentence)

    else:
        raise ValueError(f"Unknown filler condition: {filler_condition}")

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


def parse_base_filename(base_path):
    """
    Expected filename format:
    item_001_2000_base.txt
    item_001_5000_base.txt
    """
    match = re.match(r"^(item_\d+)_(\d+)_base$", base_path.stem)

    if not match:
        raise ValueError(
            f"Unexpected base document filename: {base_path.name}. "
            "Expected format like item_001_2000_base.txt"
        )

    base_item_id = match.group(1)
    word_count_target = int(match.group(2))

    return base_item_id, word_count_target


def get_distractors_for_fact(fact, all_distractors):
    target_label = fact["target_label"]
    correct_answer = fact["correct_answer"]

    filtered = [
        distractor
        for distractor in all_distractors
        if distractor["label"] != target_label
        and distractor["answer"] != correct_answer
    ]

    return filtered

def select_distractors_for_fact(fact, all_distractors, k, rng):
    target_item_id = fact["item_id"]
    target_label = fact["target_label"]
    correct_answer = fact["correct_answer"]

    same_nato_root_distractors = [
        distractor
        for distractor in all_distractors
        if distractor.get("source_item_id") == target_item_id
        and distractor["label"] != target_label
        and distractor["answer"] != correct_answer
    ]

    other_distractors = [
        distractor
        for distractor in all_distractors
        if distractor.get("source_item_id") != target_item_id
        and distractor["label"] != target_label
        and distractor["answer"] != correct_answer
    ]

    if not same_nato_root_distractors:
        raise ValueError(
            f"No same-root distractors found for {target_item_id} / {target_label}."
        )

    if len(other_distractors) < k - 1:
        raise ValueError(
            f"Need at least {k - 1} other distractors, "
            f"but only found {len(other_distractors)}."
        )

    selected = []

    # Force at least one same-NATO-root distractor.
    selected.append(rng.choice(same_nato_root_distractors))

    # Fill the remaining distractor slots with other distractors.
    selected.extend(rng.sample(other_distractors, k=k - 1))

    # Shuffle so the same-root distractor is not always in the first marker.
    rng.shuffle(selected)

    return selected

def main():
    facts = read_jsonl("data/facts.jsonl")
    distractors = read_jsonl("data/distractors.jsonl")

    base_paths = sorted(BASE_DOCS_DIR.glob("*_base.txt"))

    if not base_paths:
        raise FileNotFoundError(f"No base documents found in {BASE_DOCS_DIR}")

    print(f"Found {len(base_paths)} base documents.")
    print(f"Found {len(facts)} facts.")
    print(f"Found {len(distractors)} distractors.")

    dataset_rng = random.Random(DATASET_RANDOM_SEED)

    output_rows = []

    for base_path in base_paths:
        source_base_item_id, word_count_target = parse_base_filename(base_path)

        base_text = base_path.read_text(encoding="utf-8")
        validate_base_document(base_text, base_path)

        # Randomly choose one fact from the 25 for this base document.
        fact = dataset_rng.choice(facts)

        valid_distractors = get_distractors_for_fact(fact, distractors)

        for position_name, selected_marker in MARKERS.items():
            for filler_condition in ["generic", "distractor"]:
                condition_seed = (
                    f"{DATASET_RANDOM_SEED}_"
                    f"{base_path.stem}_"
                    f"{fact['item_id']}_"
                    f"{position_name}_"
                    f"{filler_condition}"
                )
                condition_rng = random.Random(condition_seed)

                document_text = build_document(
                    base_text=base_text,
                    answer_sentence=fact["answer_sentence"],
                    selected_marker=selected_marker,
                    filler_condition=filler_condition,
                    distractors=valid_distractors,
                    rng=condition_rng,
                    fact=fact,
                )

                doc_id = (
                    f"{source_base_item_id}_{word_count_target}_"
                    f"{fact['item_id']}_{filler_condition}_{position_name}"
                )

                output_rows.append(
                    {
                        "doc_id": doc_id,

                        # Base document info
                        "source_base_item_id": source_base_item_id,
                        "base_document_path": str(base_path),
                        "word_count_target": word_count_target,

                        # Randomly selected target fact info
                        "item_id": fact["item_id"],
                        "fact_item_id": fact["item_id"],
                        "target_label": fact["target_label"],
                        "question": fact["question"],
                        "correct_answer": fact["correct_answer"],
                        "answer_sentence": fact["answer_sentence"],

                        # Experimental variables
                        "answer_position": position_name,
                        "filler_condition": filler_condition,

                        # Final generated document
                        "document_text": document_text,
                    }
                )

    write_jsonl(OUTPUT_PATH, output_rows)

    print(f"Wrote {len(output_rows)} final documents to {OUTPUT_PATH}")

    expected_count = len(base_paths) * len(MARKERS) * 2
    print(f"Expected {expected_count} documents.")

    if len(output_rows) != expected_count:
        raise ValueError(
            f"Unexpected dataset size: got {len(output_rows)}, expected {expected_count}."
        )


if __name__ == "__main__":
    main()