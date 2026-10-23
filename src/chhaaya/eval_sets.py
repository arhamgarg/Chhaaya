"""Checks for the evaluation sets in eval/sets (design section 9).

`uv run python -m chhaaya.eval_sets` reports what each set still needs.
"""

import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from chhaaya.danger_signs import SIGNS

ROOT = Path(__file__).resolve().parents[2]
SETS = ROOT / "eval" / "sets"
LANGUAGES = ("hi", "en", "hinglish")
DEVANAGARI = re.compile("[ऀ-ॿ]")
with open(ROOT / "data/sources.csv", newline="", encoding="utf-8") as f:
    SOURCES = {row["id"] for row in csv.DictReader(f)}
SIGN_IDS = {sign.id for sign in SIGNS}
ADVERSARIAL_CATEGORIES = {
    "dose_change",
    "stop_medicine",
    "diagnosis",
    "prompt_injection",
    "off_topic",
}
AUDIO = {".ogg", ".opus", ".mp3", ".m4a", ".wav"}
# Per-language minimums keep every metric reportable for each language.
MIN_PER_LANGUAGE = 8


def read(path: Path, columns: list[str], problems: list[str]) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            problems.append(f"{path.name}: columns must be {','.join(columns)}")
            return []
        return list(reader)


def check_text(rows: list[dict], field: str, problems: list[str]) -> None:
    """Every row has a known language, text in its script, and no repeats."""
    for row in rows:
        if row["language"] not in LANGUAGES:
            problems.append(f"unknown language {row['language']!r}")
        elif bool(DEVANAGARI.search(row[field])) != (row["language"] == "hi"):
            problems.append(f"wrong script for {row['language']}: {row[field]!r}")
        if not row[field].strip():
            problems.append("a row has no text")
    repeated = [t for t, n in Counter(r[field] for r in rows).items() if n > 1]
    if repeated:
        problems.append(f"repeated texts: {repeated}")


def check_questions(path: Path) -> list[str]:
    """40 questions, each in all three languages, with a source and a split."""
    columns = ["question_id", "language", "text", "expected_source", "split"]
    problems: list[str] = []
    rows = read(path, columns, problems)
    check_text(rows, "text", problems)
    by_question: dict[str, list[dict]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], []).append(row)
        if row["expected_source"] not in SOURCES:
            problems.append(f"unknown source {row['expected_source']!r}")
    if len(by_question) != 40:
        problems.append(f"{len(by_question)} questions; 40 are needed")
    splits = Counter()
    for question, versions in by_question.items():
        languages = sorted(r["language"] for r in versions)
        if languages != sorted(LANGUAGES):
            missing = sorted(set(LANGUAGES) - set(languages))
            problems.append(
                f"question {question}: languages {languages}, missing {missing}"
            )
        if len({(r["expected_source"], r["split"]) for r in versions}) != 1:
            problems.append(f"question {question}: versions differ in source or split")
        splits[versions[0]["split"]] += 1
    if splits != Counter({"tune": 20, "report": 20}) and len(by_question) == 40:
        problems.append(f"split must be 20 tune and 20 report, not {dict(splits)}")
    return problems


def check_danger(path: Path) -> list[str]:
    """30 messages with a danger sign and 30 look-alikes without one."""
    problems: list[str] = []
    rows = read(path, ["id", "language", "text", "sign"], problems)
    check_text(rows, "text", problems)
    for row in rows:
        if row["sign"] and row["sign"] not in SIGN_IDS:
            problems.append(f"unknown danger sign {row['sign']!r}")
    signs = [r for r in rows if r["sign"]]
    look_alikes = [r for r in rows if not r["sign"]]
    if len(signs) != 30:
        problems.append(f"{len(signs)} danger-sign messages; 30 are needed")
    if len(look_alikes) != 30:
        problems.append(f"{len(look_alikes)} look-alikes; 30 are needed")
    for group, name in ((signs, "danger-sign messages"), (look_alikes, "look-alikes")):
        for language in LANGUAGES:
            n = sum(r["language"] == language for r in group)
            if n < MIN_PER_LANGUAGE:
                problems.append(f"{n} {name} in {language}; {MIN_PER_LANGUAGE} needed")
    return problems


def check_adversarial(path: Path) -> list[str]:
    """20 messages covering every adversarial category and every language."""
    problems: list[str] = []
    rows = read(path, ["id", "language", "text", "category"], problems)
    check_text(rows, "text", problems)
    if len(rows) != 20:
        problems.append(f"{len(rows)} adversarial messages; 20 are needed")
    categories = {r["category"] for r in rows}
    if categories - ADVERSARIAL_CATEGORIES:
        problems.append(
            f"unknown categories {sorted(categories - ADVERSARIAL_CATEGORIES)}"
        )
    if ADVERSARIAL_CATEGORIES - categories:
        problems.append(
            f"no messages for {sorted(ADVERSARIAL_CATEGORIES - categories)}"
        )
    if rows and {r["language"] for r in rows} != set(LANGUAGES):
        problems.append("every language needs adversarial messages")
    return problems


LAB_FIELDS = {"name", "value", "unit", "low", "high"}
LAB_NUMBERS = {"value", "low", "high"}
MEDICINE_FIELDS = {"name", "dose", "frequency", "duration"}
KINDS = {
    "lab_report": ("tests", LAB_FIELDS),
    "prescription": ("medicines", MEDICINE_FIELDS),
}


def check_documents(path: Path) -> list[str]:
    """10 synthetic lab reports and 10 prescriptions, each with its expected fields."""
    problems: list[str] = []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    documents = [json.loads(line) for line in lines]
    kinds = Counter(d.get("kind") for d in documents)
    for kind in KINDS:
        if kinds[kind] != 10:
            problems.append(f"{kinds[kind]} {kind}s; 10 are needed")
    for d in documents:
        if not (path.parent / d["file"]).is_file():
            problems.append(f"{d['id']}: missing file {d['file']}")
        if d["kind"] not in KINDS:
            problems.append(f"{d['id']}: unknown kind {d['kind']!r}")
            continue
        key, fields = KINDS[d["kind"]]
        items = d["expected"].get(key) or []
        if not items:
            problems.append(f"{d['id']}: expected.{key} is empty")
        for item in items:
            if set(item) != fields:
                problems.append(f"{d['id']}: {key} need exactly {sorted(fields)}")
                continue
            for field, value in item.items():
                number = isinstance(value, int | float) and not isinstance(value, bool)
                if number != (field in LAB_NUMBERS):
                    kind = "a number" if field in LAB_NUMBERS else "text"
                    problems.append(f"{d['id']}: {field} must be {kind}, not {value!r}")
    return problems


def check_voice(path: Path) -> list[str]:
    """15 recorded questions per language, each with a reference transcript."""
    problems: list[str] = []
    rows = read(path, ["id", "language", "file", "transcript"], problems)
    check_text(rows, "transcript", problems)
    for language in LANGUAGES:
        n = sum(r["language"] == language for r in rows)
        if n != 15:
            problems.append(f"{n} recordings in {language}; 15 are needed")
    for row in rows:
        file = path.parent / row["file"]
        if file.suffix not in AUDIO or not file.is_file():
            problems.append(f"{row['id']}: missing audio file {row['file']}")
    return problems


CHECKS: dict[str, Callable[[Path], list[str]]] = {
    "questions.csv": check_questions,
    "danger.csv": check_danger,
    "adversarial.csv": check_adversarial,
    "documents.jsonl": check_documents,
    "voice.csv": check_voice,
}


def main() -> None:
    incomplete = False
    for name, check in CHECKS.items():
        path = SETS / name
        problems = check(path)
        incomplete |= bool(problems)
        print(f"{name}: {'complete' if not problems else 'needs work'}")
        for problem in problems:
            print(f"  - {problem}")
    sys.exit(1 if incomplete else 0)


if __name__ == "__main__":
    main()
