"""Offline metrics, alignment, faithful SHAP, and persisted-state boundaries.

Failure cases: unequal/unknown labels, malformed probabilities, repeated or
misaligned baseline rows, missing/corrupt/stale reports and caches, explanations
that do not reconstruct all four probabilities, and cache misses after changes.
"""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from chhaaya import model_metrics as metrics
from chhaaya.intent_data import LABELS


def test_metrics_on_imbalanced_confusion_and_raw_probability_auc():
    expected = [LABELS[i] for i in [0, 0, 0, 1, 1, 2, 3]]
    predicted = [LABELS[i] for i in [0, 0, 1, 1, 2, 2, 0]]
    # Probabilities rank every class perfectly, despite different routed labels.
    probabilities = np.full((7, 4), 0.05)
    for i, label in enumerate(expected):
        probabilities[i, LABELS.index(label)] = 0.85
    result = metrics.calculate_metrics(expected, predicted, probabilities)
    assert result["confusion_matrix"] == [
        [2, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ]
    assert result["accuracy"] == pytest.approx(4 / 7)
    assert result["precision_macro"] == pytest.approx(5 / 12)
    assert result["precision_weighted"] == pytest.approx(0.5)
    assert result["recall_macro"] == pytest.approx(13 / 24)
    assert result["recall_weighted"] == pytest.approx(4 / 7)
    assert result["f1_macro"] == pytest.approx(11 / 24)
    assert result["f1_weighted"] == pytest.approx(11 / 21)
    assert result["roc_auc_macro"] == 1
    assert result["roc_auc_weighted"] == 1
    hard_only = metrics.calculate_metrics(expected, predicted)
    assert hard_only["roc_auc_macro"] is None
    assert hard_only["roc_auc_weighted"] is None


@pytest.mark.parametrize(
    "probabilities", [np.ones((4, 3)), np.ones((4, 4)), np.full((4, 4), float("nan"))]
)
def test_metrics_reject_invalid_probabilities(probabilities):
    with pytest.raises(ValueError, match="probabilit"):
        metrics.calculate_metrics(list(LABELS), list(LABELS), probabilities)


def test_baseline_pass_one_requires_exact_row_alignment(tmp_path):
    rows = [{"text": "a", "label": "question", "language": "en"}]
    path = tmp_path / "results.csv"
    fields = ["model", "run", "index", "language", "expected", "predicted"]
    records = [
        dict(zip(fields, [model, run, 0, "en", "question", label], strict=True))
        for model in ["keywords", "sarvam-105b"]
        for run, label in [(1, "question"), (2, "smalltalk")]
    ]

    def write():
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    write()
    assert metrics.baseline_predictions(rows, path) == {
        "keywords": ["question"],
        "sarvam-105b": ["question"],
    }
    records[0]["expected"] = "smalltalk"
    write()
    with pytest.raises(ValueError, match="align"):
        metrics.baseline_predictions(rows, path)
    records[0]["expected"] = "question"
    records.append(records[0])
    write()
    with pytest.raises(ValueError, match="duplicate"):
        metrics.baseline_predictions(rows, path)


def test_partition_shap_all_classes_reconstruct_and_cache(tmp_path, monkeypatch):
    import shap

    def probabilities(texts):
        output = []
        for text in texts:
            scores = np.array(
                [1 + text.count(word) for word in ["pain", "remind", "taken", "hello"]],
                dtype=float,
            )
            output.append(scores / scores.sum())
        return np.array(output)

    engine = object.__new__(metrics.ExplanationEngine)
    engine.digest = "test-model-digest"
    monkeypatch.setattr(engine, "predict_proba", probabilities)
    monkeypatch.setattr(
        engine,
        "classifier",
        SimpleNamespace(
            tokenizer=lambda text, **kwargs: {"offset_mapping": [(0, len(text))]}
        ),
        raising=False,
    )
    monkeypatch.setattr(metrics, "_checkpoint", lambda: {"sha256": engine.digest})
    engine.threshold = 0.8
    engine.explainer = shap.Explainer(
        probabilities,
        shap.maskers.Text(r"\W"),
        algorithm="partition",
        output_names=list(LABELS),
        seed=42,
    )
    monkeypatch.setattr(metrics, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(metrics, "_get_engine", lambda: engine)
    text = "hello remind pain pain taken"
    np.random.seed(321)
    expected_random = np.random.RandomState(321).random()
    first = metrics.explain_message(text)
    assert np.random.random() == expected_random
    assert first["predicted"] == "question"
    assert first["routed"] == "question"
    assert first["id"] is None
    assert first["language"] == "custom"
    np.testing.assert_allclose(
        np.array(first["base_values"]) + np.array(first["values"]).sum(axis=0),
        first["probabilities"],
        atol=1e-5,
    )
    assert np.abs(first["values"]).sum() > 0
    assert len(first["values"][0]) == 4
    assert len(list(tmp_path.glob("*.json"))) == 1
    monkeypatch.setattr(
        engine, "explainer", lambda *a, **k: pytest.fail("cache should avoid inference")
    )
    assert metrics.explain_message(text) == first
    cache = next(tmp_path.glob("*.json"))
    original_cache = cache.read_text()
    saved = json.loads(original_cache)
    saved["example"]["values"][0][0] += 0.5
    cache.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="reconstruct"):
        metrics.explain_message(text)
    cache.write_text(original_cache)
    engine.digest = "changed-model"
    with pytest.raises(BaseException, match="cache should avoid inference"):
        metrics.explain_message(text)


@pytest.mark.parametrize("text", ["", "   ", "word " * 3000])
def test_custom_explanations_reject_empty_or_unbounded_text(text):
    with pytest.raises(ValueError, match="text"):
        metrics.explain_message(text)


def test_report_rejects_missing_corrupt_and_stale_without_model_load(
    tmp_path, monkeypatch
):
    path = tmp_path / "evaluation.json"
    monkeypatch.setattr(metrics, "REPORT_PATH", path)
    monkeypatch.setattr(metrics, "_get_engine", lambda: pytest.fail("model loaded"))
    with pytest.raises(FileNotFoundError, match="model_metrics"):
        metrics.load_report()
    path.write_text("{broken")
    with pytest.raises(ValueError, match="corrupt"):
        metrics.load_report()
    # The real report is checked separately after the offline CLI completes.


def test_example_validation_checks_all_classes_and_routing():
    example = dict(
        id=None,
        text="remind me",
        language="custom",
        expected=None,
        predicted="reminder_setup",
        routed="question",
        confidence=0.7,
        probabilities=[0.1, 0.7, 0.1, 0.1],
        base_values=[0.25] * 4,
        tokens=["remind", " me"],
        values=[[-0.1, 0.2, -0.05, -0.05], [-0.05, 0.25, -0.1, -0.1]],
    )
    metrics.validate_example(example)
    example["routed"] = "reminder_setup"
    with pytest.raises(ValueError, match="rout"):
        metrics.validate_example(example)


@pytest.fixture
def report_fixture(tmp_path, monkeypatch):
    """A portable report that needs neither model weights nor a provider."""
    import hashlib

    data = tmp_path / "data/intent"
    data.mkdir(parents=True)
    source = metrics.DATA_DIR / "test.csv"
    (data / "test.csv").write_bytes(source.read_bytes())
    rows = metrics.load_labelled(data / "test.csv")
    baseline_metadata = dict(
        test_sha256=metrics.file_sha256(data / "test.csv"), f1_pass=1, runs=3
    )
    (data / "results.json").write_text(json.dumps(baseline_metadata))
    monkeypatch.setattr(metrics, "ROOT", tmp_path)
    monkeypatch.setattr(metrics, "DATA_DIR", data)
    monkeypatch.setattr(metrics, "REPORT_PATH", data / "model_evaluation.json")
    hashes = {
        "train.csv": "train-hash",
        "test.csv": metrics.file_sha256(data / "test.csv"),
        "results.csv": "baseline-hash",
    }
    monkeypatch.setattr(metrics, "_source_hashes", lambda: hashes)
    monkeypatch.setattr(metrics, "_checkpoint", lambda: {"sha256": "model-hash"})
    monkeypatch.setattr(metrics, "_get_engine", lambda: pytest.fail("model loaded"))
    examples = []
    for i, row in enumerate(rows):
        probabilities = [0.01] * 4
        probabilities[LABELS.index(row["label"])] = 0.97
        examples.append(
            dict(
                id=i,
                text=row["text"],
                language=row["language"],
                expected=row["label"],
                predicted=row["label"],
                routed=row["label"],
                confidence=0.97,
                probabilities=probabilities,
                base_values=[0.25] * 4,
                tokens=[row["text"]],
                values=[[value - 0.25 for value in probabilities]],
            )
        )
    models = []
    for name in ["muril", "keywords", "sarvam-105b"]:

        def score(selected, name=name):
            labels = [row["label"] for row in selected]
            scores = [
                [0.97 if label == row["label"] else 0.01 for label in LABELS]
                for row in selected
            ]
            return metrics.calculate_metrics(
                labels, labels, scores if name == "muril" else None
            )

        models.append(
            dict(
                id=name,
                name=name,
                kind="trained" if name == "muril" else "baseline",
                metrics=dict(
                    overall=score(rows),
                    by_language={
                        language: score(
                            [row for row in rows if row["language"] == language]
                        )
                        for language in metrics.LANGUAGES
                    },
                ),
                roc_auc_reason=None if name == "muril" else "Hard labels only.",
                predictions=[row["label"] for row in rows],
            )
        )
    figures = {
        name: f"eval/figures/intent-shap-{name}.png" for name in ["global", "local"]
    }
    for relative in figures.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture-figure")
    report = dict(
        schema_version=1,
        labels=list(LABELS),
        test=dict(
            rows=120,
            sha256=hashes["test.csv"],
            languages={"en": 40, "hi": 40, "hinglish": 40},
        ),
        models=models,
        explainability=dict(
            model_id="muril",
            method="SHAP Partition",
            reason="Test SHAP.",
            sample_size=120,
            seed=42,
            units="class probability",
            examples=examples,
            global_figure=figures["global"],
            local_figure=figures["local"],
            worked_example_id=0,
            **{"global": [{"token": "a", "importance": 0.1}]},
        ),
        provenance=dict(
            source_sha256=hashes,
            baseline_metadata=baseline_metadata,
            checkpoint=dict(sha256="model-hash"),
            explanation_settings=metrics.SETTINGS,
            figure_sha256={
                name: hashlib.sha256(b"fixture-figure").hexdigest() for name in figures
            },
        ),
    )
    return report, hashes


def test_report_reads_without_inference_and_rejects_stale_hashes(report_fixture):
    report, hashes = report_fixture
    metrics._write_json(metrics.REPORT_PATH, report)
    assert metrics.load_report() == report
    hashes["results.csv"] = "different-pass-one-records"
    with pytest.raises(ValueError, match="stale"):
        metrics.load_report()


@pytest.mark.parametrize("corruption", ["language", "nan", "example", "figure"])
def test_report_corruption_fails_clearly(report_fixture, corruption):
    report, _ = report_fixture
    if corruption == "language":
        report["models"][0]["metrics"]["by_language"] = {}
    elif corruption == "nan":
        report["models"][0]["metrics"]["overall"]["accuracy"] = float("nan")
    elif corruption == "example":
        report["explainability"]["examples"][0]["values"][0][3] += 0.1
    else:
        (metrics.ROOT / report["explainability"]["global_figure"]).write_bytes(
            b"changed"
        )
    metrics.REPORT_PATH.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="corrupt|stale"):
        metrics.load_report()


def test_file_checksum_invalidates_after_replacement(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"first checkpoint")
    first = metrics.file_sha256(path)
    path.write_bytes(b"different checkpoint")
    assert metrics.file_sha256(path) != first


@pytest.mark.parametrize(
    "expected,predicted", [([], []), (["question"], []), (["unknown"], ["question"])]
)
def test_metrics_reject_mismatched_or_unknown_labels(expected, predicted):
    with pytest.raises(ValueError, match="label"):
        metrics.calculate_metrics(expected, predicted)


def test_baseline_metadata_rejects_a_different_test_set(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "DATA_DIR", tmp_path)
    (tmp_path / "test.csv").write_text("held-out bytes")
    path = tmp_path / "results.json"
    path.write_text(json.dumps(dict(test_sha256="wrong-checksum", f1_pass=1, runs=3)))
    with pytest.raises(ValueError, match="baseline.*checksum"):
        metrics._baseline_metadata()
    path.write_text(
        json.dumps(
            dict(
                test_sha256=metrics.file_sha256(tmp_path / "test.csv"),
                f1_pass=1,
                runs=3,
            )
        )
    )
    assert metrics._baseline_metadata()["f1_pass"] == 1
