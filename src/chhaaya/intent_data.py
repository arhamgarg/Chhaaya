"""Loading and checking the intent datasets (design section 7)."""

import csv
import sys
from collections import Counter
from pathlib import Path

LABELS = ("question", "reminder_setup", "reminder_reply", "smalltalk")
LANGUAGES = ("hi", "en", "hinglish")


def load_labelled(path: Path) -> list[dict[str, str]]:
    """Read text,label,language rows; fail on an unknown value or a repeated text."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["label"] not in LABELS:
            raise ValueError(f"unknown label {row['label']!r} for {row['text']!r}")
        if row["language"] not in LANGUAGES:
            raise ValueError(f"unknown language {row['language']!r}")
    texts = Counter(row["text"] for row in rows)
    repeated = [text for text, n in texts.items() if n > 1]
    if repeated:
        raise ValueError(f"duplicate texts: {repeated}")
    return rows


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Agreement between two labellers beyond what chance would give (1 is perfect)."""
    if len(a) != len(b) or not a:
        raise ValueError("both labellers must label the same, non-empty list")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    count_a, count_b = Counter(a), Counter(b)
    chance = sum(count_a[label] * count_b[label] for label in count_a) / n**2
    if chance == 1:
        raise ValueError("kappa is undefined when both labellers use only one label")
    return (observed - chance) / (1 - chance)


def load_agreement(path: Path) -> dict[str, str]:
    """Read one labeller's id,label file; fail on an unknown label or a repeated id."""
    labels: dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["label"] not in LABELS:
                raise ValueError(f"unknown label {row['label']!r} for id {row['id']!r}")
            if row["id"] in labels:
                raise ValueError(f"repeated id {row['id']!r}")
            labels[row["id"]] = row["label"]
    return labels


def main() -> None:
    """Print kappa for two id,label files given on the command line."""
    first, second = (load_agreement(Path(path)) for path in sys.argv[1:3])
    if first.keys() != second.keys():
        raise SystemExit("the two files must label the same ids")
    ids = sorted(first, key=int)
    kappa = cohens_kappa([first[i] for i in ids], [second[i] for i in ids])
    agree = sum(first[i] == second[i] for i in ids)
    print(f"{agree}/{len(ids)} labels agree, Cohen's kappa {kappa:.2f}")


if __name__ == "__main__":
    main()
