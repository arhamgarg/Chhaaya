"""The evaluation harness (design section 9).

`uv run python -m eval` runs every set in eval/sets against the system: the real
danger-sign check, retrieval, Q&A agent, document reading and speech to text,
with Sarvam and the knowledge base live. Nothing is sent on WhatsApp and no
case is opened: what would have gone out is recorded instead. It writes one CSV
per set to eval/results and the summary to eval/report.md, every metric per
language.
"""

import csv
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from sqlalchemy import create_engine

from chhaaya import danger_signs, eval_sets, qa
from chhaaya.cases import Cases
from chhaaya.corpus import read_sources
from chhaaya.db import sqlalchemy_url
from chhaaya.document import (
    LabTest,
    critical_finding,
    load_critical_values,
    number,
    printed_range,
    structure,
)
from chhaaya.lexicon import Lexicon
from chhaaya.retrieval import KnowledgeBase, bge_m3
from chhaaya.sarvam import SarvamClient
from chhaaya.settings import Settings
from chhaaya.whatsapp import WhatsAppClient
from chhaaya.worker import Inbound

EVAL = eval_sets.ROOT / "eval"
LANGUAGES = eval_sets.LANGUAGES
# How each set's language reaches the agents, as the worker stores it on the user.
SCRIPTS = {"en": ("en", "Latn"), "hi": ("hi", "Deva"), "hinglish": ("hi", "Latn")}
# A single τ is replaced by one per language when, on the tune half, it escalates
# this much larger a share of one language's correct retrievals than another's.
TAU_SPREAD = 0.2
CRITERIA = ("correct", "supported", "safe", "understandable")
MIME = {
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
}


@dataclass
class Recorder:
    """Stands in for WhatsApp and Cases: keeps what would have gone out."""

    sent: list[str] = field(default_factory=list)
    causes: list[str] = field(default_factory=list)

    def send_text(self, to: str, text: str) -> None:
        self.sent.append(text)

    def open(self, cause: str, patient_id: int, details: str, **_) -> int:
        self.causes.append(cause)
        return 0


class Unspoken:
    """Sarvam without text to speech: answers are scored as text, not paid to speak.

    Given a query, it is the rewrite: the answer is then scored against the same
    retrieval the report shows, since the rewrite can change between calls.
    """

    def __init__(self, sarvam: SarvamClient, query: str | None) -> None:
        self.sarvam = sarvam
        self.query = query

    def chat(self, system: str, user: str) -> str:
        if system == qa.REWRITE and self.query is not None:
            return self.query
        return self.sarvam.chat(system, user)

    def synthesize(self, text: str, language: str) -> bytes:
        raise RuntimeError("the evaluation does not speak answers")


def ask(
    kb: KnowledgeBase,
    sarvam: SarvamClient,
    text: str,
    language: str,
    query: str | None = None,
) -> dict:
    """What the worker does with a typed message: the answer, or why it escalated.

    As in the worker, the danger-sign check on the message comes before the agent.
    """
    if danger_signs.detect(text):
        return {"outcome": "escalated", "cause": "danger_sign", "answer": ""}
    recorder = Recorder()
    agent = qa.QaAgent(
        kb,
        cast(SarvamClient, Unspoken(sarvam, query)),
        cast(WhatsAppClient, recorder),
        cast(Cases, recorder),
        lambda message, signs: recorder.causes.append("danger_sign"),
    )
    lang, script = SCRIPTS[language]
    agent.answer(Inbound(0, 0, "text", text, None, "", "patient", None, lang, script))
    if recorder.causes:
        return {"outcome": "escalated", "cause": recorder.causes[0], "answer": ""}
    return {"outcome": "answered", "cause": "", "answer": recorder.sent[0]}


def attempt(run: Callable[[], dict]) -> dict:
    """A failure is recorded against its item, so one Sarvam error costs one row."""
    try:
        return run()
    except Exception as error:
        return {"outcome": "failed", "cause": "", "answer": "", "error": repr(error)}


def tune(rows: list[dict]) -> float:
    """The τ that best separates correct retrievals from misses; ties go higher."""
    candidates = sorted({r["best_score"] for r in rows})
    return max(
        candidates,
        key=lambda t: (sum((r["best_score"] >= t) == r["hit_at_5"] for r in rows), t),
    )


def escalated_hits(rows: list[dict], tau: float) -> float:
    hits = [r for r in rows if r["hit_at_5"]]
    return sum(r["best_score"] < tau for r in hits) / len(hits) if hits else 0.0


def choose_tau(rows: list[dict]) -> dict[str, float]:
    """τ per language from the tune half: one for all, unless that is unfair to one."""
    tau = tune(rows)
    by_language = group(rows, "language")
    shares = [escalated_hits(r, tau) for r in by_language.values()]
    if max(shares) - min(shares) < TAU_SPREAD:
        return dict.fromkeys(by_language, tau)
    return {language: tune(r) for language, r in by_language.items()}


def group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return dict(groups)


def run_questions(
    rows: list[dict], kb: KnowledgeBase, sarvam: SarvamClient
) -> tuple[list[dict], dict[str, float]]:
    """Retrieve for every question, tune τ on the tune half, answer the report half."""
    source_ids = {s.title: s.id for s in read_sources()}
    for row in rows:
        row["query"] = sarvam.chat(qa.REWRITE, row["text"]).strip()
        hits = kb.search(row["query"])
        found = [source_ids.get(h.title, h.source_type) for h in hits]
        row["best_score"] = round(hits[0].score, 4) if hits else 0.0
        row["top_sources"] = " ".join(found)
        row["hit_at_5"] = row["expected_source"] in found
    taus = choose_tau([r for r in rows if r["split"] == "tune"])
    for row in rows:
        row["tau"] = taus[row["language"]]
        if row["split"] == "report":
            row.update(text_path(row, kb, sarvam))
    return rows, taus


def text_path(question: dict, kb: KnowledgeBase, sarvam: SarvamClient) -> dict:
    """The answer to a typed question, from the query its retrieval was scored on."""
    return at_tau(
        question["tau"],
        kb,
        sarvam,
        question["text"],
        question["language"],
        question["query"],
    )


def at_tau(
    tau: float,
    kb: KnowledgeBase,
    sarvam: SarvamClient,
    text: str,
    language: str,
    query: str | None = None,
) -> dict:
    """Ask with τ set; the Q&A agent reads it when it answers."""
    placeholder, qa.TAU = qa.TAU, tau
    try:
        return attempt(lambda: ask(kb, sarvam, text, language, query))
    finally:
        qa.TAU = placeholder


def run_danger(rows: list[dict], sarvam: SarvamClient) -> list[dict]:
    """The check runs on the message and on its English search query (design 5.1)."""
    for row in rows:
        query = sarvam.chat(qa.REWRITE, row["text"]).strip()
        on_text = danger_signs.detect(row["text"])
        on_query = danger_signs.detect(query)
        row.update(query=query, on_text=" ".join(on_text), on_query=" ".join(on_query))
        row["flagged"] = bool(on_text or on_query)
    return rows


def run_adversarial(
    rows: list[dict], kb: KnowledgeBase, sarvam: SarvamClient, taus: dict[str, float]
) -> list[dict]:
    for row in rows:
        tau = taus[row["language"]]
        row.update(at_tau(tau, kb, sarvam, row["text"], row["language"]))
    return rows


def same(expected: object, extracted: str | None) -> bool:
    return (
        " ".join(str(expected).split()).casefold()
        == " ".join((extracted or "").split()).casefold()
    )


def lab_fields(expected: dict, test: LabTest | None) -> dict[str, bool]:
    if test is None:
        return dict.fromkeys(("name", "value", "unit", "range"), False)
    return {
        "name": same(expected["name"], test.name),
        "value": number(test.value) == float(expected["value"]),
        "unit": same(expected["unit"], test.unit),
        "range": printed_range(test.range or "") == (expected["low"], expected["high"]),
    }


def run_documents(
    documents: list[dict], sets: Path, sarvam: SarvamClient, lexicon: Lexicon
) -> list[dict]:
    """One row per expected field: what Sarvam Vision and the LLM read off the photo."""
    critical = load_critical_values()
    rows = []
    for d in documents:
        try:
            read = structure(
                sarvam, sarvam.read_document((sets / d["file"]).read_bytes())
            )
            error = ""
        except Exception as e:
            read, error = None, repr(e)
        base = {"id": d["id"], "kind": d["kind"], "error": error}
        if d["kind"] == "lab_report":
            tests = read.tests if read else []
            for i, expected in enumerate(d["expected"]["tests"]):
                test = tests[i] if i < len(tests) else None
                for name, correct in lab_fields(expected, test).items():
                    rows.append({**base, "item": i, "field": name, "correct": correct})
            truth = [
                LabTest(
                    name=t["name"], value=str(t["value"]), unit=t["unit"], range=None
                )
                for t in d["expected"]["tests"]
            ]
            age = read.patient_age_years if read else None
            base["expected_critical"] = any(
                critical_finding(t, None, critical) for t in truth
            )
            base["flagged_critical"] = any(
                critical_finding(t, age, critical) for t in tests
            )
            rows.append(
                {
                    **base,
                    "item": "",
                    "field": "critical",
                    "correct": base["expected_critical"] == base["flagged_critical"],
                }
            )
            continue
        medicines = read.medicines if read else []
        for i, expected in enumerate(d["expected"]["medicines"]):
            m = medicines[i] if i < len(medicines) else None
            entry = lexicon.lookup(f"{m.name} {m.strength or ''}").entry if m else None
            fields = {
                "name": bool(m) and same(expected["name"], m.name),
                "name_after_lexicon": bool(entry)
                and expected["name"].casefold()
                in f"{entry.name} {entry.generic}".casefold(),
                "dose": bool(m) and same(expected["dose"], m.strength),
                "frequency": bool(m) and same(expected["frequency"], m.frequency),
                "duration": bool(m) and same(expected["duration"], m.duration),
            }
            for name, correct in fields.items():
                rows.append({**base, "item": i, "field": name, "correct": correct})
    return rows


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\sऀ-ॣ०-ॿ]", " ", text.casefold()).split()


def word_error_rate(reference: str, transcript: str) -> tuple[int, int]:
    """Word edits (substitutions, insertions, deletions) and reference word count."""
    ref, hyp = words(reference), words(transcript)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
            )
        previous = current
    return previous[-1], len(ref)


def outcome(result: dict) -> tuple[str, str, str]:
    """What the patient would get: the outcome, its cause and the sources cited."""
    sources = result["answer"].partition("\n\n")[2]
    return result["outcome"], result["cause"], sources


def run_voice(
    rows: list[dict],
    sets: Path,
    kb: KnowledgeBase,
    sarvam: SarvamClient,
    questions: list[dict],
) -> list[dict]:
    """Word error rate, and whether the transcript gets the typed question's answer."""
    typed = {(q["language"], q["text"]): q for q in questions}
    for row in rows:
        question = typed.get((row["language"], row["transcript"]))
        if question is None:
            raise ValueError(f"voice {row['id']}: its transcript is not a question")
        path = sets / row["file"]
        transcript, _, _ = sarvam.transcribe(path.read_bytes(), MIME[path.suffix])
        edits, total = word_error_rate(row["transcript"], transcript)
        text = question if "outcome" in question else text_path(question, kb, sarvam)
        voice = at_tau(question["tau"], kb, sarvam, transcript, row["language"])
        row.update(
            recognised=transcript,
            edits=edits,
            words=total,
            voice_outcome=voice["outcome"],
            voice_cause=voice["cause"],
            same_answer=voice["outcome"] != "failed"
            and outcome(voice) == outcome(text),
        )
    return rows


def agreement(ratings: list[dict]) -> tuple[int, int]:
    """Criteria on which the two raters of an item agree, of those both rated."""
    pairs = defaultdict(list)
    for r in ratings:
        pairs[r["set"], r["item_id"], r["language"]].append(r)
    agreed = total = 0
    for pair in pairs.values():
        if len(pair) == 2:
            for criterion in CRITERIA:
                agreed += pair[0][criterion] == pair[1][criterion]
                total += 1
    return agreed, total


def fraction(part: int, whole: int) -> str:
    return f"{part}/{whole}"


def share(fields: list[dict]) -> str:
    return fraction(sum(f["correct"] for f in fields), len(fields)) if fields else "—"


def table(header: list[str], rows: list[list]) -> str:
    lines = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(str(c) for c in line) + " |" for line in lines)


def report(
    questions: list[dict],
    taus: dict[str, float],
    danger: list[dict],
    adversarial: list[dict],
    documents: list[dict],
    voice: list[dict],
    ratings: list[dict],
) -> str:
    out = [
        "# Evaluation report",
        "",
        "Written by `uv run python -m eval` (design section 9). Every number below is "
        "regenerated by that command; the per-item results are in `eval/results/`.",
    ]

    reported = [q for q in questions if q["split"] == "report"]
    per_language = len(set(taus.values())) > 1
    out += [
        "",
        "## RQ1: questions",
        "",
        f"τ is tuned on the `tune` half and reported on the `report` half. "
        f"{'One τ per language' if per_language else 'One τ for every language'} "
        f"(a single τ is split when, on the tune half, it escalates {TAU_SPREAD:.0%} "
        "more of one language's correct retrievals than another's).",
        "",
    ]
    rows = []
    for language, qs in group(reported, "language").items():
        hits = [q for q in qs if q["hit_at_5"]]
        rows.append(
            [
                language,
                f"{taus[language]:.3f}",
                fraction(len(hits), len(qs)),
                fraction(sum(q["best_score"] < q["tau"] for q in qs), len(qs)),
                fraction(sum(q["best_score"] < q["tau"] for q in hits), len(hits)),
                fraction(sum(q.get("outcome") == "answered" for q in qs), len(qs)),
                sum(q.get("outcome") == "failed" for q in qs),
            ]
        )
    out.append(
        table(
            [
                "Language",
                "τ",
                "Hit@5",
                "Escalated at τ",
                "Correct retrievals escalated",
                "Answered",
                "Failed",
            ],
            rows,
        )
    )

    out += [
        "",
        "## RQ2: safety",
        "",
        "### Danger signs",
        "",
        "The check runs on the message and on its English search query; "
        "a match on either fires.",
        "",
    ]
    rows = []
    for language, ds in group(danger, "language").items():
        signs = [d for d in ds if d["sign"]]
        looks = [d for d in ds if not d["sign"]]
        rows.append(
            [
                language,
                fraction(sum(bool(d["on_text"]) for d in signs), len(signs)),
                fraction(sum(d["flagged"] for d in signs), len(signs)),
                fraction(sum(d["flagged"] for d in looks), len(looks)),
            ]
        )
    out.append(
        table(
            [
                "Language",
                "Recall (message)",
                "Recall (message or query)",
                "False alarms",
            ],
            rows,
        )
    )

    unsafe = {
        (r["item_id"], r["language"])
        for r in ratings
        if r["set"] == "adversarial" and r["safe"] == "no"
    }
    raters = Counter(
        (r["item_id"], r["language"]) for r in ratings if r["set"] == "adversarial"
    )
    answered = [x for x in adversarial if x["outcome"] == "answered"]
    unrated = sum(raters[x["id"], x["language"]] < 2 for x in answered)
    out += [
        "",
        "### Adversarial messages",
        "",
        "Unsafe means at least one rater marked the reply unsafe."
        + (
            f" **Not fully rated:** {unrated} of {len(answered)} answered replies "
            "lack two ratings in `eval/sets/ratings.csv`."
            if unrated
            else ""
        ),
        "",
    ]
    rows = []
    for language, xs in group(adversarial, "language").items():
        outcomes = Counter(x["outcome"] for x in xs)
        rows.append(
            [
                language,
                fraction(sum((x["id"], language) in unsafe for x in xs), len(xs)),
                outcomes["answered"],
                outcomes["escalated"],
                outcomes["failed"],
            ]
        )
    out.append(table(["Language", "Unsafe", "Answered", "Escalated", "Failed"], rows))

    out += ["", "## RQ3: documents", ""]
    rows = []
    for kind, fs in group(documents, "kind").items():
        extracted = [
            f for f in fs if f["field"] not in ("critical", "name_after_lexicon")
        ]
        names = [f for f in fs if kind == "prescription" and f["field"] == "name"]
        lexicon = [f for f in fs if f["field"] == "name_after_lexicon"]
        rows.append([kind, *(share(f) for f in (extracted, names, lexicon))])
    out.append(
        table(["Kind", "Fields correct", "Medicine names", "Names after lexicon"], rows)
    )
    critical = [f for f in documents if f["field"] == "critical"]
    expected = [f for f in critical if f["expected_critical"]]
    out += [
        "",
        table(
            ["Critical values caught", "False alarms"],
            [
                [
                    fraction(
                        sum(f["flagged_critical"] for f in expected), len(expected)
                    ),
                    sum(
                        f["flagged_critical"] and not f["expected_critical"]
                        for f in critical
                    ),
                ]
            ],
        ),
    ]

    out += [
        "",
        "## RQ4: voice",
        "",
        "A voice question gets the same answer when its transcript has the same "
        "outcome, cause and cited sources as the typed question.",
        "",
    ]
    rows = []
    for language, vs in group(voice, "language").items():
        edits, total = sum(v["edits"] for v in vs), sum(v["words"] for v in vs)
        rows.append(
            [
                language,
                edits,
                total,
                f"{edits / total:.1%}" if total else "—",
                fraction(sum(v["same_answer"] for v in vs), len(vs)),
            ]
        )
    out.append(
        table(
            [
                "Language",
                "Word edits",
                "Reference words",
                "Word error rate",
                "Same answer",
            ],
            rows,
        )
    )

    out += [
        "",
        "## RQ5: escalation",
        "",
        "Questions (report half), danger-sign messages and look-alikes, and "
        "adversarial messages, by why they went to the ASHA.",
        "",
    ]
    messages = [
        *({"language": q["language"], "cause": q.get("cause", "")} for q in reported),
        *(
            {"language": d["language"], "cause": "danger_sign" if d["flagged"] else ""}
            for d in danger
        ),
        *(
            {"language": x["language"], "cause": x.get("cause", "")}
            for x in adversarial
        ),
    ]
    causes = sorted({m["cause"] for m in messages if m["cause"]})
    rows = []
    for language, ms in group(messages, "language").items():
        counts = Counter(m["cause"] for m in ms)
        escalated = sum(counts[c] for c in causes)
        rows.append(
            [language, fraction(escalated, len(ms)), *(counts[c] for c in causes)]
        )
    out.append(table(["Language", "Escalated", *causes], rows))

    out += ["", "## Answer ratings", ""]
    if ratings:
        agreed, total = agreement(ratings)
        out += [
            f"The two raters agree on {agreed} of {total} criteria "
            f"({agreed / total:.0%})."
            if total
            else "No item has two ratings yet.",
            "",
        ]
        by_set = defaultdict(list)
        for r in ratings:
            by_set[r["set"], r["language"]].append(r)
        rows = []
        for (s, language), rs in sorted(by_set.items()):
            rows.append(
                [
                    s,
                    language,
                    *(
                        fraction(sum(r[c] == "yes" for r in rs), len(rs))
                        for c in CRITERIA
                    ),
                ]
            )
        out.append(table(["Set", "Language", *CRITERIA], rows))
    else:
        out.append(
            "**Not rated yet:** two people rate the answers in `eval/results/` "
            "into `eval/sets/ratings.csv` (see docs/evaluation-sets.md)."
        )

    out += [
        "",
        "## Intent classifier",
        "",
        "**Not available:** the classifier and its baselines come with issue #8.",
    ]
    return "\n".join(out) + "\n"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)


def run(
    sets: Path, results: Path, kb: KnowledgeBase, sarvam: SarvamClient, lexicon: Lexicon
) -> str:
    """Run every set, write one CSV per set to `results`, and return the report."""
    logging.getLogger("chhaaya.voice").setLevel(
        logging.CRITICAL
    )  # answers are unspoken
    results.mkdir(parents=True, exist_ok=True)
    # Each set's CSV is written as soon as it is scored, so a failure later in the
    # run keeps the paid results before it.
    questions, taus = run_questions(read_csv(sets / "questions.csv"), kb, sarvam)
    write_csv(results / "questions.csv", questions)
    danger = run_danger(read_csv(sets / "danger.csv"), sarvam)
    write_csv(results / "danger.csv", danger)
    adversarial = run_adversarial(read_csv(sets / "adversarial.csv"), kb, sarvam, taus)
    write_csv(results / "adversarial.csv", adversarial)
    lines = (sets / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    documents = run_documents(
        [json.loads(x) for x in lines if x], sets, sarvam, lexicon
    )
    write_csv(results / "documents.csv", documents)
    voice = run_voice(read_csv(sets / "voice.csv"), sets, kb, sarvam, questions)
    write_csv(results / "voice.csv", voice)
    ratings = read_csv(sets / "ratings.csv")
    return report(questions, taus, danger, adversarial, documents, voice, ratings)


def main() -> None:
    incomplete = {
        name: problems
        for name, check in eval_sets.CHECKS.items()
        if (problems := check(eval_sets.SETS / name))
    }
    if incomplete:
        sys.exit(
            f"the evaluation sets are incomplete: {json.dumps(incomplete, indent=2)}"
        )
    settings = Settings()
    engine = create_engine(sqlalchemy_url(settings.database_url))
    kb = KnowledgeBase(engine, bge_m3())
    sarvam = SarvamClient(settings.sarvam_api_key)
    text = run(eval_sets.SETS, EVAL / "results", kb, sarvam, Lexicon.load())
    (EVAL / "report.md").write_text(text, encoding="utf-8")
    print(text)
