"""Offline checkpoint metrics and probability-space SHAP for the intent classifier.

Run ``uv run python -m chhaaya.model_metrics`` to regenerate the report and figures.
No provider, database, environment secrets, training, or app worker is involved.
"""

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

from chhaaya.intent_data import LABELS, LANGUAGES, load_labelled

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/intent"
MODEL_DIR = DATA_DIR / "model"
REPORT_PATH = DATA_DIR / "model_evaluation.json"
CACHE_DIR = MODEL_DIR / "explanations"
CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "router.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training.json",
)
SEED = 42
MAX_EVALS = 128
BATCH_SIZE = 16
MAX_TEXT_CHARS = 4000
THRESHOLD = 0.8
TOLERANCE = 2e-4
SETTINGS = {
    "seed": SEED,
    "max_evals": MAX_EVALS,
    "batch_size": BATCH_SIZE,
    "algorithm": "partition",
    "masker": "shap.maskers.Text",
    "mask_token": "[MASK]",
    "collapse_mask_token": False,
    "max_length": 64,
    "confidence_threshold": THRESHOLD,
    "output": "softmax class probability",
    "link": "identity",
    "fixed_context": None,
    "rng": "numpy legacy seed 42 per explanation, caller state restored",
}


@lru_cache(maxsize=64)
def _hash_at_stat(path: Path, size: int, modified: int) -> str:
    del size, modified  # Cache invalidation includes the file's size and mtime.
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_sha256(path: Path) -> str:
    stat = path.stat()
    return _hash_at_stat(path, stat.st_size, stat.st_mtime_ns)


def _source_hashes() -> dict[str, str]:
    return {
        name: file_sha256(DATA_DIR / name)
        for name in ("train.csv", "test.csv", "results.csv", "results.json")
    }


def _checkpoint() -> dict[str, Any]:
    hashes = {name: file_sha256(MODEL_DIR / name) for name in CHECKPOINT_FILES}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    training = json.loads((MODEL_DIR / "training.json").read_text())
    if training["train_sha256"] != file_sha256(DATA_DIR / "train.csv"):
        raise ValueError("checkpoint training-data checksum differs from train.csv")
    router = json.loads((MODEL_DIR / "router.json").read_text())
    if router != {"max_length": 64, "confidence_threshold": THRESHOLD}:
        raise ValueError("checkpoint must use 64 tokens and the deployed 0.8 threshold")
    return {"sha256": digest, "files": hashes, "training": training, "router": router}


def calculate_metrics(
    expected: list[str], predicted: list[str], probabilities: Any = None
) -> dict[str, Any]:
    """Routed-label metrics; OvR AUC uses raw probabilities in LABELS order."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    if not expected or len(expected) != len(predicted):
        raise ValueError(
            "expected and predicted labels must have the same nonzero size"
        )
    if any(label not in LABELS for label in expected + predicted):
        raise ValueError("unknown intent label")
    result: dict[str, Any] = {
        "n": len(expected),
        "accuracy": float(accuracy_score(expected, predicted)),
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=list(LABELS)
        ).tolist(),
        "roc_auc_macro": None,
        "roc_auc_weighted": None,
    }
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            expected, predicted, labels=list(LABELS), average=average, zero_division=0
        )
        result.update(
            {
                f"precision_{average}": float(precision),
                f"recall_{average}": float(recall),
                f"f1_{average}": float(f1),
            }
        )
    if probabilities is not None:
        scores = np.asarray(probabilities, dtype=float)
        if (
            scores.shape != (len(expected), 4)
            or not np.isfinite(scores).all()
            or (scores < 0).any()
            or (scores > 1).any()
            or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)
        ):
            raise ValueError("probabilities must be finite normalized n-by-4 scores")
        if set(expected) != set(LABELS):
            raise ValueError("multiclass ROC requires all four expected labels")
        # Integer targets preserve checkpoint label order (sklearn sorts strings).
        target = [LABELS.index(label) for label in expected]
        for average in ("macro", "weighted"):
            result[f"roc_auc_{average}"] = float(
                roc_auc_score(
                    target,
                    scores,
                    multi_class="ovr",
                    average=average,
                    labels=list(range(4)),
                )
            )
    return result


def _baseline_metadata() -> dict[str, Any]:
    metadata = json.loads((DATA_DIR / "results.json").read_text())
    if metadata["test_sha256"] != file_sha256(DATA_DIR / "test.csv"):
        raise ValueError("baseline test checksum differs from the held-out CSV")
    if metadata["f1_pass"] != 1 or metadata["runs"] < 1:
        raise ValueError("baseline metadata must identify pass 1")
    return {key: metadata[key] for key in ("test_sha256", "f1_pass", "runs")}


def baseline_predictions(
    rows: list[dict[str, str]], path: Path
) -> dict[str, list[str]]:
    """Use exactly pass 1, rejecting duplicate, missing or misaligned records."""
    with path.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    output = {}
    for name in ("keywords", "sarvam-105b"):
        selected = [r for r in records if r["model"] == name and r["run"] == "1"]
        indices = [int(r["index"]) for r in selected]
        if len(set(indices)) != len(indices):
            raise ValueError(f"duplicate pass-1 baseline indices for {name}")
        if indices != list(range(len(rows))):
            raise ValueError(f"baseline {name} does not align with held-out row order")
        for saved, row in zip(selected, rows, strict=True):
            if (
                saved["expected"] != row["label"]
                or saved["language"] != row["language"]
                or saved["predicted"] not in LABELS
            ):
                raise ValueError(f"baseline {name} does not align with test.csv")
        output[name] = [r["predicted"] for r in selected]
    return output


def _vector(value: Any, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(isinstance(x, (int, float)) and math.isfinite(x) for x in value)
    )


def validate_example(example: dict[str, Any]) -> None:
    """Reject corrupt serialization, non-additive values and wrong routing."""
    required = {
        "id",
        "text",
        "language",
        "expected",
        "predicted",
        "routed",
        "confidence",
        "probabilities",
        "base_values",
        "tokens",
        "values",
    }
    if not isinstance(example, dict) or not required <= example.keys():
        raise ValueError("corrupt explanation: required example fields are missing")
    probabilities, base = example["probabilities"], example["base_values"]
    tokens, values = example["tokens"], example["values"]
    if (
        not _vector(probabilities, 4)
        or not _vector(base, 4)
        or any(x < 0 or x > 1 for x in probabilities)
        or abs(sum(probabilities) - 1) > 1e-5
        or not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(token, str) for token in tokens)
        or not isinstance(values, list)
        or len(values) != len(tokens)
        or any(not _vector(value, 4) for value in values)
    ):
        raise ValueError("corrupt explanation probability or attribution dimensions")
    for i in range(4):
        if (
            abs(base[i] + sum(value[i] for value in values) - probabilities[i])
            > TOLERANCE
        ):
            raise ValueError("SHAP values do not reconstruct all four probabilities")
    predicted = LABELS[max(range(4), key=lambda i: probabilities[i])]
    confidence = max(probabilities)
    if (
        example["predicted"] != predicted
        or not isinstance(example["confidence"], (int, float))
        or not math.isfinite(example["confidence"])
        or abs(example["confidence"] - confidence) > 1e-5
    ):
        raise ValueError("corrupt explanation prediction or confidence")
    routed = predicted if confidence >= THRESHOLD else "question"
    if example["routed"] != routed:
        raise ValueError("explanation routing disagrees with the 0.8 fallback")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publication keeps a concurrent reader from seeing a partial report.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            # Remove partial output on interrupts as well as serialization failures.
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


class ExplanationEngine:
    """A private CPU checkpoint instance, independent of the application worker."""

    def __init__(self) -> None:
        import shap

        from chhaaya.intent import Classifier

        checkpoint = _checkpoint()
        self.digest = checkpoint["sha256"]
        self.classifier = Classifier(MODEL_DIR)
        self.threshold = self.classifier.config.confidence_threshold
        # Identity link explains probabilities, rather than logits or fallback rules.
        self.explainer = shap.Explainer(
            self.predict_proba,
            shap.maskers.Text(
                self.classifier.tokenizer,
                mask_token="[MASK]",
                collapse_mask_token=False,
            ),
            algorithm="partition",
            output_names=list(LABELS),
            seed=SEED,
        )

    def predict_proba(self, texts: Any) -> Any:
        import numpy as np
        import torch

        texts = [str(text) for text in texts]
        outputs = []
        for start in range(0, len(texts), BATCH_SIZE):
            encoded = self.classifier.tokenizer(
                texts[start : start + BATCH_SIZE],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            with torch.inference_mode():
                outputs.append(
                    self.classifier.model(**encoded)
                    .logits.softmax(dim=-1)
                    .cpu()
                    .numpy()
                )
        return np.concatenate(outputs, axis=0)

    def explain(self, text: str) -> dict[str, Any]:
        import numpy as np

        # Bound the masker too. Otherwise a long input could create thousands of
        # features even though the model sees only the first 64 tokens.
        encoded = self.classifier.tokenizer(
            text, truncation=True, max_length=64, return_offsets_mapping=True
        )
        end = max(offset[1] for offset in encoded["offset_mapping"])
        model_text = text[:end]
        # Partition uses numpy's legacy RNG for priority-queue ties. Its
        # constructor does not forward Explainer(seed=...), so seed each call.
        state = np.random.get_state()
        np.random.seed(SEED)
        try:
            result = self.explainer(
                [model_text], max_evals=MAX_EVALS, batch_size=BATCH_SIZE, silent=True
            )[0]
        finally:
            np.random.set_state(state)
        probabilities = self.predict_proba([text])[0]
        # The original and masker-visible prefix must represent the same model input.
        np.testing.assert_allclose(
            self.predict_proba([model_text])[0], probabilities, atol=1e-5
        )
        predicted = LABELS[int(probabilities.argmax())]
        confidence = float(probabilities.max())
        example = {
            "id": None,
            "text": text,
            "language": "custom",
            "expected": None,
            "predicted": predicted,
            "routed": predicted if confidence >= self.threshold else "question",
            "confidence": confidence,
            "probabilities": probabilities.tolist(),
            "base_values": np.asarray(result.base_values).tolist(),
            "tokens": [str(token) for token in result.data],
            "values": np.asarray(result.values).tolist(),
        }
        validate_example(example)
        return example


@lru_cache(maxsize=1)
def _get_engine() -> ExplanationEngine:
    return ExplanationEngine()


def _cached_example(text: str, engine: ExplanationEngine) -> dict[str, Any]:
    identity = {
        "text": text,
        "model_sha256": engine.digest,
        "settings": SETTINGS,
        "shap_version": version("shap"),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"corrupt explanation cache {path}: {error}") from error
        if not isinstance(cached, dict) or cached.get("identity") != identity:
            raise ValueError(f"corrupt or stale explanation cache {path}")
        if "example" not in cached:
            raise ValueError(f"corrupt explanation cache {path}: missing example")
        example = cached["example"]
        validate_example(example)
        if example["text"] != text:
            raise ValueError(f"corrupt explanation cache text in {path}")
        return example
    example = engine.explain(text)
    _write_json(path, {"identity": identity, "example": example})
    return example


def explain_message(text: str) -> dict[str, Any]:
    """Lazy, bounded local inference with an on-disk text/checkpoint SHAP cache.

    Call serially from the UI's threadpool. This explains raw classifier output;
    it does not explain the confidence threshold or the pending-reminder guard.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text must contain 1 to {MAX_TEXT_CHARS} nonempty characters")
    engine = _get_engine()
    # Detect replacement after the engine was loaded instead of serving old weights.
    if _checkpoint()["sha256"] != engine.digest:
        raise ValueError("checkpoint changed; restart the metrics backend")
    return _cached_example(text, engine)


def load_report() -> dict[str, Any]:
    """Read and validate the artifact without loading a model or running inference."""
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"missing {REPORT_PATH}; run uv run python -m chhaaya.model_metrics"
        )
    try:
        report = json.loads(REPORT_PATH.read_text())
        if report["schema_version"] != 1 or report["labels"] != list(LABELS):
            raise ValueError("unsupported schema or label order")
        provenance = report["provenance"]
        if provenance["baseline_metadata"] != _baseline_metadata():
            raise ValueError("stale baseline metadata")
        if (
            provenance["source_sha256"] != _source_hashes()
            or provenance["checkpoint"]["sha256"] != _checkpoint()["sha256"]
            or provenance["explanation_settings"] != SETTINGS
        ):
            raise ValueError("stale report: source, checkpoint or settings changed")
        rows = load_labelled(DATA_DIR / "test.csv")
        if report["test"] != {
            "rows": len(rows),
            "sha256": file_sha256(DATA_DIR / "test.csv"),
            "languages": dict(Counter(row["language"] for row in rows)),
        }:
            raise ValueError("stale report: held-out dataset differs")
        if [model["id"] for model in report["models"]] != [
            "muril",
            "keywords",
            "sarvam-105b",
        ]:
            raise ValueError("report must compare MuRIL and both baselines")
        for model in report["models"]:
            if set(model["metrics"]["by_language"]) != set(LANGUAGES):
                raise ValueError("incomplete per-language metrics")
            if model["kind"] != ("trained" if model["id"] == "muril" else "baseline"):
                raise ValueError("invalid model kind")
            if model["id"] != "muril" and not model["roc_auc_reason"]:
                raise ValueError("missing reason for unavailable baseline ROC-AUC")
            for language, metric in [
                (None, model["metrics"]["overall"]),
                *model["metrics"]["by_language"].items(),
            ]:
                n = (
                    len(rows)
                    if language is None
                    else report["test"]["languages"][language]
                )
                matrix = metric["confusion_matrix"]
                if (
                    metric["n"] != n
                    or len(matrix) != 4
                    or any(len(row) != 4 for row in matrix)
                    or any(type(x) is not int or x < 0 for row in matrix for x in row)
                    or sum(map(sum, matrix)) != n
                ):
                    raise ValueError("invalid metric count or confusion matrix")
                for name in (
                    "accuracy",
                    "precision_macro",
                    "precision_weighted",
                    "recall_macro",
                    "recall_weighted",
                    "f1_macro",
                    "f1_weighted",
                    "roc_auc_macro",
                    "roc_auc_weighted",
                ):
                    value = metric[name]
                    if name.startswith("roc_auc") and model["id"] != "muril":
                        if value is not None:
                            raise ValueError("hard-label baseline ROC-AUC must be null")
                        continue
                    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                        raise ValueError(f"invalid metric {name}")
        xai = report["explainability"]
        if (
            xai["model_id"] != "muril"
            or xai["method"] != "SHAP Partition"
            or xai["seed"] != SEED
            or xai["units"] != "class probability"
            or not 0 <= xai["worked_example_id"] < len(rows)
            or not xai["global"]
            or xai["sample_size"] != len(rows)
            or len(xai["examples"]) != len(rows)
        ):
            raise ValueError("incomplete SHAP report")
        for item in xai["global"]:
            if (
                not isinstance(item["token"], str)
                or not isinstance(item["importance"], (int, float))
                or not math.isfinite(item["importance"])
                or item["importance"] < 0
            ):
                raise ValueError("invalid global token importance")
        for index, (example, row) in enumerate(zip(xai["examples"], rows, strict=True)):
            validate_example(example)
            if (
                example["id"] != index
                or example["text"] != row["text"]
                or example["expected"] != row["label"]
                or example["language"] != row["language"]
            ):
                raise ValueError("SHAP examples do not align with held-out rows")
        for name in ("global", "local"):
            path = ROOT / xai[f"{name}_figure"]
            if file_sha256(path) != provenance["figure_sha256"][name]:
                raise ValueError(f"stale {name} figure")
        return report
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as error:
        raise ValueError(
            f"corrupt or stale metrics report {REPORT_PATH}: {error}"
        ) from error


def _global_importance(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Sum repeated subword occurrences per example, then average absolute class
    # contributions over all four outputs and all 120 rows, including absences.
    importance: dict[str, float] = defaultdict(float)
    for example in examples:
        for token, values in zip(example["tokens"], example["values"], strict=True):
            if token.strip():
                importance[token.strip()] += sum(abs(value) for value in values) / 4
    return [
        {"token": token, "importance": value / len(examples)}
        for token, value in sorted(importance.items(), key=lambda x: (-x[1], x[0]))
    ]


def _figures(report: dict[str, Any]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt
    from matplotlib.ft2font import FT2Font

    xai = report["explainability"]
    candidates = (
        "Arial Unicode MS",
        "Noto Sans Devanagari",
        "Kohinoor Devanagari",
        "Devanagari Sangam MN",
    )
    fonts = {font.name: font.fname for font in font_manager.fontManager.ttflist}
    devanagari = next((name for name in candidates if name in fonts), None)
    if devanagari is None:
        raise RuntimeError(
            "Install a Devanagari font to render the multilingual figures"
        )
    plt.rcParams.update(
        {
            "font.family": [devanagari, "DejaVu Sans"],
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    coverage = set(FT2Font(fonts[devanagari]).get_charmap())
    # Reject empty glyphs rather than saving a misleading publication artifact.
    shown = "".join(item["token"] for item in xai["global"][:20])
    example = xai["examples"][xai["worked_example_id"]]
    shown += example["text"]
    if any(ord(char) not in coverage for char in shown if not char.isspace()):
        raise RuntimeError(f"{devanagari} lacks a glyph used in the figures")
    paths = {name: ROOT / xai[f"{name}_figure"] for name in ("global", "local")}
    paths["global"].parent.mkdir(parents=True, exist_ok=True)
    top = list(reversed(xai["global"][:20]))
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    ax.barh(
        [item["token"] for item in top],
        [item["importance"] for item in top],
        color="#286d89",
    )
    ax.set(
        xlabel="Mean absolute SHAP contribution (class probability)",
        title="MuRIL intent classifier · global token importance",
    )
    fig.suptitle(
        "120 held-out messages · all four classes · Partition SHAP", fontsize=11
    )
    fig.savefig(paths["global"], dpi=240)
    plt.close(fig)
    selected = [i for i, token in enumerate(example["tokens"]) if token.strip()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for index, ax in enumerate(axes.flat):
        values = [example["values"][i][index] for i in selected]
        ax.barh(
            range(len(selected)),
            values,
            color=["#286d89" if value >= 0 else "#bb553d" for value in values],
        )
        ax.set_yticks(
            range(len(selected)), [example["tokens"][i].strip() for i in selected]
        )
        ax.invert_yaxis()
        ax.axvline(0, color="#888888", linewidth=0.8)
        ax.set(
            title=f"{LABELS[index]}  |  base {example['base_values'][index]:.3f}"
            f" + contributions = {example['probabilities'][index]:.3f}",
            xlabel="SHAP contribution (class probability)",
        )
    fig.suptitle(
        f"MuRIL · held-out example {example['id']}\n{example['text']}\n"
        f"Raw: {example['predicted']} · routed: {example['routed']}"
        " · explains probabilities, not clinical safety",
        fontsize=12,
    )
    fig.savefig(paths["local"], dpi=240)
    plt.close(fig)
    return {name: file_sha256(path) for name, path in paths.items()}


def main() -> None:
    import torch

    # Bound CPU parallelism only in this standalone offline process.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    rows = load_labelled(DATA_DIR / "test.csv")
    languages = dict(Counter(row["language"] for row in rows))
    if len(rows) != 120 or languages != {"en": 40, "hi": 40, "hinglish": 40}:
        raise ValueError("expected the fixed 120-row held-out set, 40 per language")
    sources = _source_hashes()
    checkpoint = _checkpoint()
    baseline_metadata = _baseline_metadata()
    baselines = baseline_predictions(rows, DATA_DIR / "results.csv")
    engine = _get_engine()
    examples = []
    for index, row in enumerate(rows):
        example = dict(_cached_example(row["text"], engine))
        example.update(id=index, language=row["language"], expected=row["label"])
        examples.append(example)
        print(f"SHAP {index + 1}/120 ({row['language']})", flush=True)
    predictions = {"muril": [example["routed"] for example in examples], **baselines}
    probabilities = [example["probabilities"] for example in examples]
    models: list[dict[str, Any]] = []
    for name, predicted in predictions.items():

        def score(
            indices: list[int], predicted: list[str] = predicted, name: str = name
        ) -> dict[str, Any]:
            return calculate_metrics(
                [rows[i]["label"] for i in indices],
                [predicted[i] for i in indices],
                [probabilities[i] for i in indices] if name == "muril" else None,
            )

        models.append(
            {
                "id": name,
                "name": {
                    "muril": "Fine-tuned MuRIL (900 training rows)",
                    "keywords": "Keyword rules",
                    "sarvam-105b": "Sarvam-105B zero-shot",
                }[name],
                "kind": "trained" if name == "muril" else "baseline",
                "metrics": {
                    "overall": score(list(range(len(rows)))),
                    "by_language": {
                        language: score(
                            [
                                i
                                for i, row in enumerate(rows)
                                if row["language"] == language
                            ]
                        )
                        for language in LANGUAGES
                    },
                },
                "roc_auc_reason": None
                if name == "muril"
                else "Saved pass-1 labels have no probabilities.",
                "prediction_source": "cached CPU checkpoint inference"
                if name == "muril"
                else "data/intent/results.csv, run=1 only",
                "predictions": predicted,
            }
        )
    report = {
        "schema_version": 1,
        "test": {
            "rows": len(rows),
            "sha256": sources["test.csv"],
            "languages": languages,
        },
        "labels": list(LABELS),
        "models": models,
        "explainability": {
            "model_id": "muril",
            "method": "SHAP Partition",
            "reason": "Partition SHAP supports the saved BERT/MuRIL tokenizer with a "
            "Text masker and a custom softmax-probability function. All four "
            "outputs are explained. Bounded evaluations approximate token "
            "allocation while preserving probability additivity.",
            "sample_size": len(rows),
            "seed": SEED,
            "units": "class probability",
            "global": _global_importance(examples),
            "examples": examples,
            "global_figure": "eval/figures/intent-shap-global.png",
            "local_figure": "eval/figures/intent-shap-local.png",
            "worked_example_id": next(
                i
                for i, row in enumerate(rows)
                if row["language"] == "hi" and row["label"] == "reminder_setup"
            ),
            "aggregation": "Sum absolute token contributions across occurrences and "
            "all four classes, divided by 4 classes and all 120 messages. "
            "Tokens include subwords. Absences count as zero.",
            "scope": "Explains raw classifier probabilities, not the 0.8 fallback, the "
            "pending-reminder guard, danger detection, or clinical safety.",
        },
        "provenance": {
            "source_sha256": sources,
            "checkpoint": checkpoint,
            "versions": {
                package: version(package)
                for package in (
                    "torch",
                    "transformers",
                    "tokenizers",
                    "shap",
                    "numpy",
                    "scikit-learn",
                    "matplotlib",
                )
            },
            "explanation_settings": SETTINGS,
            "metrics_predictions": "MuRIL argmax is routed to question below 0.8. "
            "No pending-reminder state is assumed for any model in this benchmark.",
            "roc_auc": "Multiclass one-vs-rest, macro and support-weighted, from "
            "raw MuRIL probabilities in labels order, before confidence fallback.",
            "baseline_metadata": baseline_metadata,
            "baseline_pass": 1,
            "external_calls": False,
            "retrained": False,
            "clinical_safety_evaluated": False,
            "command": "uv run python -m chhaaya.model_metrics",
        },
    }
    report["provenance"]["figure_sha256"] = _figures(report)
    # Reject a source/checkpoint replacement during a long run.
    if sources != _source_hashes() or checkpoint != _checkpoint():
        raise ValueError("sources or checkpoint changed during evaluation")
    _write_json(MODEL_DIR / "evaluation.json", report)
    _write_json(REPORT_PATH, report)
    load_report()
    for model in models:
        metric = model["metrics"]["overall"]
        print(
            f"{model['id']}: n={metric['n']} accuracy={metric['accuracy']:.6f} "
            f"macro-F1={metric['f1_macro']:.6f}",
            flush=True,
        )
    print(f"Saved {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
