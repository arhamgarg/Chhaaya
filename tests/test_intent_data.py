import csv
import re
from collections import Counter
from pathlib import Path

import pytest

from chhaaya.intent_data import LABELS, cohens_kappa, load_agreement, load_labelled

DATA = Path(__file__).parents[1] / "data/intent"
DEVANAGARI = re.compile("[ऀ-ॿ]")


def test_training_set_is_big_balanced_and_in_the_right_script():
    rows = load_labelled(DATA / "train.csv")

    counts = Counter((row["language"], row["label"]) for row in rows)
    assert {language for language, _ in counts} == {"hi", "en", "hinglish"}
    for language in ("hi", "en", "hinglish"):
        total = sum(n for (lang, _), n in counts.items() if lang == language)
        assert total >= 300
        # Roughly balanced: no label has less than 20% or more than 30% of a language.
        for label in LABELS:
            assert 0.2 <= counts[language, label] / total <= 0.3
    for row in rows:
        has_devanagari = bool(DEVANAGARI.search(row["text"]))
        assert has_devanagari == (row["language"] == "hi"), row["text"]


def test_test_set_is_forty_per_language_and_shares_nothing_with_training():
    test = load_labelled(DATA / "test.csv")
    train = {row["text"].strip().lower() for row in load_labelled(DATA / "train.csv")}

    counts = Counter((row["language"], row["label"]) for row in test)
    for language in ("hi", "en", "hinglish"):
        assert sum(n for (lang, _), n in counts.items() if lang == language) == 40
        assert all(counts[language, label] >= 5 for label in LABELS)
    assert not [row["text"] for row in test if row["text"].strip().lower() in train]
    for row in test:
        assert bool(DEVANAGARI.search(row["text"])) == (row["language"] == "hi")


def test_a_repeated_message_is_rejected(tmp_path: Path):
    path = tmp_path / "train.csv"
    path.write_text("text,label,language\nhello,smalltalk,en\nhello,smalltalk,en\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_labelled(path)


def test_an_unknown_label_is_rejected(tmp_path: Path):
    path = tmp_path / "train.csv"
    path.write_text("text,label,language\nhello,greeting,en\n")
    with pytest.raises(ValueError, match="greeting"):
        load_labelled(path)


def test_agreement_sample_is_fifty_training_messages_without_labels():
    sample = list(csv.DictReader(open(DATA / "agreement_sample.csv")))
    train = {row["text"] for row in load_labelled(DATA / "train.csv")}

    assert len(sample) == 50
    assert list(sample[0]) == ["id", "text"]
    assert {row["text"] for row in sample} <= train


def test_kappa_is_one_for_identical_labels_and_zero_for_chance_agreement():
    labels = ["question", "reminder_setup", "reminder_reply", "smalltalk"] * 3
    assert cohens_kappa(labels, labels) == 1.0

    # Agreement exactly as often as chance predicts, so kappa is 0.
    a = ["question", "question", "smalltalk", "smalltalk"]
    b = ["question", "smalltalk", "question", "smalltalk"]
    assert cohens_kappa(a, b) == 0.0

    # 4 of 5 agree; chance agreement is (3*4 + 2*1) / 25 = 0.56.
    a = ["question"] * 3 + ["smalltalk"] * 2
    b = ["question"] * 3 + ["smalltalk", "question"]
    assert cohens_kappa(a, b) == pytest.approx((0.8 - 0.56) / (1 - 0.56))


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ("1,question\n2,Question\n", "Question"),
        ("1,question\n2,question \n", "question "),
        ("1,question\n2,\n", "''"),
        ("1,question\n1,smalltalk\n", "repeated id '1'"),
    ],
)
def test_an_agreement_file_with_a_bad_label_or_repeated_id_is_rejected(
    tmp_path: Path, rows: str, error: str
):
    path = tmp_path / "labeller.csv"
    path.write_text("id,label\n" + rows)
    with pytest.raises(ValueError, match=re.escape(error)):
        load_agreement(path)


def test_kappa_is_undefined_when_both_labellers_use_one_label():
    with pytest.raises(ValueError, match="undefined"):
        cohens_kappa(["question"] * 3, ["question"] * 3)
