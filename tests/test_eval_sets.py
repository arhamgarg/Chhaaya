import csv
import json
from pathlib import Path

from chhaaya import eval_sets
from chhaaya.eval_sets import (
    check_adversarial,
    check_danger,
    check_documents,
    check_questions,
    check_voice,
)

TEXT = {
    "hi": "मुझे बुखार है {}",
    "en": "I have fever {}",
    "hinglish": "mujhe bukhar hai {}",
}


def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def questions(tmp_path: Path, drop: int = 0, source: str = "asha_module_7") -> Path:
    rows = [
        [q, lang, TEXT[lang].format(q), source, "tune" if q < 20 else "report"]
        for q in range(40)
        for lang in ("hi", "en", "hinglish")
    ]
    header = ["question_id", "language", "text", "expected_source", "split"]
    return write_csv(tmp_path / "questions.csv", header, rows[drop:])


def test_a_complete_question_set_passes(tmp_path: Path):
    assert check_questions(questions(tmp_path)) == []


def test_a_question_missing_a_language_or_citing_an_unknown_source_fails(
    tmp_path: Path,
):
    problems = check_questions(questions(tmp_path, drop=1))
    assert any("question 0" in p and "hi" in p for p in problems)
    assert any("40" in p for p in check_questions(questions(tmp_path, drop=3)))
    problems = check_questions(questions(tmp_path, source="a_blog"))
    assert any("a_blog" in p for p in problems)


def test_hindi_must_be_devanagari_and_the_others_latin(tmp_path: Path):
    path = questions(tmp_path)
    rows = list(csv.reader(open(path, encoding="utf-8")))
    rows[1][2] = "mujhe bukhar hai"  # a Hindi row in the Latin script
    write_csv(path, rows[0], rows[1:])
    assert any("script" in p for p in check_questions(path))


def danger(tmp_path: Path, signs: int = 30) -> Path:
    languages = ["hi", "en", "hinglish"] * 20
    rows = [
        [n, lang, TEXT[lang].format(n), "convulsions" if n < signs else ""]
        for n, lang in enumerate(languages)
    ]
    return write_csv(tmp_path / "danger.csv", ["id", "language", "text", "sign"], rows)


def test_danger_messages_need_thirty_signs_and_thirty_look_alikes(tmp_path: Path):
    assert check_danger(danger(tmp_path)) == []
    assert any("look-alike" in p for p in check_danger(danger(tmp_path, signs=40)))


def test_a_danger_message_must_name_a_known_sign(tmp_path: Path):
    path = danger(tmp_path)
    rows = list(csv.reader(open(path, encoding="utf-8")))
    rows[1][3] = "headache"
    write_csv(path, rows[0], rows[1:])
    assert any("headache" in p for p in check_danger(path))


def test_adversarial_messages_cover_every_category(tmp_path: Path):
    categories = sorted(eval_sets.ADVERSARIAL_CATEGORIES)
    languages = ["hi", "en", "hinglish"]
    rows = [
        [n, languages[n % 3], TEXT[languages[n % 3]].format(n), categories[n % 5]]
        for n in range(20)
    ]
    header = ["id", "language", "text", "category"]
    path = write_csv(tmp_path / "adversarial.csv", header, rows)
    assert check_adversarial(path) == []

    rows = [row[:3] + ["dose_change"] for row in rows]
    path = write_csv(tmp_path / "adversarial.csv", header, rows)
    assert any("prompt_injection" in p for p in check_adversarial(path))


def documents(tmp_path: Path, broken: str = "") -> Path:
    (tmp_path / "documents").mkdir(parents=True)
    lines = []
    for n in range(20):
        kind = "lab_report" if n < 10 else "prescription"
        file = f"documents/{n}.png"
        (tmp_path / file).write_bytes(b"\x89PNG")
        if kind == "lab_report":
            expected = {
                "tests": [
                    {"name": "Haemoglobin", "value": 9.1, "unit": "g/dL", "low": 12,
                     "high": 15}
                ]
            }  # fmt: skip
        else:
            expected = {
                "medicines": [
                    {"name": "Amlodipine", "dose": "5 mg", "frequency": "once daily",
                     "duration": "30 days"}
                ]
            }  # fmt: skip
        if broken == "unit" and n == 3:
            del expected["tests"][0]["unit"]
        if broken == "value" and n == 3:
            expected["tests"][0]["value"] = "9.1"
        if broken == "kind" and n == 3:
            kind = "lab-report"
        lines.append(json.dumps({"id": str(n), "kind": kind, "file": file,
                                 "expected": expected}))  # fmt: skip
    path = tmp_path / "documents.jsonl"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_documents_need_ten_of_each_kind_with_their_fields_and_files(tmp_path: Path):
    path = documents(tmp_path)
    assert check_documents(path) == []
    (tmp_path / "documents/12.png").unlink()
    assert any("12.png" in p for p in check_documents(path))


def test_a_lab_test_without_a_unit_fails(tmp_path: Path):
    assert any("unit" in p for p in check_documents(documents(tmp_path, "unit")))


def test_a_lab_value_must_be_a_number_and_the_kind_known(tmp_path: Path):
    problems = check_documents(documents(tmp_path / "value", "value"))
    assert any("3" in p and "value" in p for p in problems)
    problems = check_documents(documents(tmp_path / "kind", "kind"))
    assert any("lab-report" in p for p in problems)


def test_voice_needs_fifteen_recordings_per_language_with_transcripts(tmp_path: Path):
    (tmp_path / "voice").mkdir()
    rows = []
    for lang in ("hi", "en", "hinglish"):
        for n in range(15):
            file = f"voice/{lang}-{n}.ogg"
            (tmp_path / file).write_bytes(b"OggS")
            rows.append([f"{lang}-{n}", lang, file, TEXT[lang].format(n)])
    header = ["id", "language", "file", "transcript"]
    path = write_csv(tmp_path / "voice.csv", header, rows)
    assert check_voice(path) == []

    path = write_csv(tmp_path / "voice.csv", header, rows[1:])
    assert any("15" in p and "hi" in p for p in check_voice(path))


def test_every_committed_set_passes_its_checks():
    for name, check in eval_sets.CHECKS.items():
        assert check(eval_sets.SETS / name) == [], name
