"""Publish saved model evidence without training or making provider calls."""

import math
from pathlib import Path

SCORES = {
    "accuracy": "Accuracy",
    "precision_macro": "P macro",
    "precision_weighted": "P weighted",
    "recall_macro": "R macro",
    "recall_weighted": "R weighted",
    "f1_macro": "F1 macro",
    "f1_weighted": "F1 weighted",
    "roc_auc_macro": "AUC macro",
    "roc_auc_weighted": "AUC weighted",
}


def comparison(data: dict, language: str = "overall") -> str:
    selected = [
        (model["name"], model["metrics"]["overall"])
        if language == "overall"
        else (model["name"], model["metrics"]["by_language"][language])
        for model in data["models"]
    ]
    best = {
        key: max(
            (row[key] for _, row in selected if row[key] is not None), default=None
        )
        for key in SCORES
    }
    lines = [
        "| Model | " + " | ".join(SCORES.values()) + " |",
        "| " + " | ".join(["---"] * (len(SCORES) + 1)) + " |",
    ]
    for name, row in selected:
        values = []
        for key in SCORES:
            value = row[key]
            rendered = "N/A" if value is None else f"{value:.3f}"
            highest = best[key]
            if (
                value is not None
                and highest is not None
                and math.isclose(value, highest, rel_tol=0, abs_tol=1e-12)
            ):
                rendered = f"**{rendered}**"
            values.append(rendered)
        lines.append("| " + " | ".join([name, *values]) + " |")
    return "\n".join(lines)


def explanations(data: dict, figures: str) -> str:
    xai = data["explainability"]
    example = next(e for e in xai["examples"] if e["id"] == xai["worked_example_id"])
    index = data["labels"].index(example["predicted"])
    tokens = [
        (token.strip(), values[index])
        for token, values in zip(example["tokens"], example["values"], strict=True)
        if token.strip()
    ]
    strongest = sorted(tokens, key=lambda item: abs(item[1]), reverse=True)[:3]
    local_features = ", ".join(
        f"“{token}” ({value:+.3f})" for token, value in strongest
    )
    global_features = ", ".join(f"“{item['token']}”" for item in xai["global"][:5])
    baseline = example["base_values"][index]
    contribution = sum(values[index] for values in example["values"])
    probability = example["probabilities"][index]
    text = example["text"].replace("\n", " ")
    return f"""## Explainability

MuRIL uses **{xai["method"]}**. {xai["reason"]} LIME is not needed.
These token attributions explain class probabilities, not medical reasoning
or the deterministic confidence fallback.

The global summary uses all {xai["sample_size"]} held-out messages because the
test set has fewer than 500 rows. Explanations are cached with the model and
test-set identity. The UI reads saved results; it never trains or evaluates
the test set on page load.

![MuRIL global token importance]({figures}/intent-shap-global.png)

The most influential tokens in the saved global summary include {global_features}.
Their importance describes how the intent classifier separates these messages;
it does not establish a clinical relationship. Local contributions depend on
the rest of the message and can change direction between examples.

### Worked local example

> {text}

Raw prediction: **{example["predicted"]}** ({probability:.3f} probability).
The deployed router returns **{example["routed"]}** after its confidence rule.
Largest token contributions for the raw predicted class: {local_features}.

Masked baseline {baseline:.4f} + token contributions {contribution:+.4f}
= {baseline + contribution:.4f}; model probability {probability:.4f}.
SHAP values are approximate.

![MuRIL local token contributions]({figures}/intent-shap-local.png)
"""


def sections(data: dict, figures: str = "figures", *, compact: bool = False) -> str:
    rows = data["test"]["rows"]
    lines = [
        "## Model performance",
        "",
        f"Held-out intent test set: {rows} messages. MuRIL is the only model trained "
        "here; keywords and Sarvam are comparison baselines. Bold marks the best "
        "score in each column, including ties. P = precision, R = recall. "
        "AUC uses one-vs-rest class probabilities; "
        "saved hard-label baselines have no AUC.",
        "",
        comparison(data),
        "",
    ]
    if compact:
        lines.extend(
            [
                "## Explainability",
                "",
                f"MuRIL uses {data['explainability']['method']} with token masking. "
                "The UI shows cached global importance and local explanations for "
                "held-out or entered text. See the [worked example and figures]"
                "(eval/report.md#explainability). "
                "These explain routing, not clinical decisions.",
                "",
            ]
        )
    else:
        for language, label in (
            ("en", "English"),
            ("hi", "Hindi"),
            ("hinglish", "Hinglish"),
        ):
            lines.extend([f"### {label}", "", comparison(data, language), ""])
        lines.extend(
            [
                "### Confusion matrices",
                "",
                "Rows are actual classes; columns are predicted classes.",
                "",
            ]
        )
        for model in data["models"]:
            lines.extend(
                [
                    f"#### {model['name']}",
                    "",
                    "| Actual / predicted | " + " | ".join(data["labels"]) + " |",
                    "| " + " | ".join(["---"] * 5) + " |",
                ]
            )
            for label, row in zip(
                data["labels"],
                model["metrics"]["overall"]["confusion_matrix"],
                strict=True,
            ):
                lines.append("| " + " | ".join([label, *map(str, row)]) + " |")
            lines.append("")
        lines.extend(
            [
                "## Dataset provenance",
                "",
                "Published intent dataset: [IEEE DataPort]"
                "(https://doi.org/10.21227/m81y-kj83). "
                "The 900 intent training examples were drafted with an LLM, then "
                "read and corrected by the team. The 120 held-out intent messages "
                "were written by people and were not used for training or choosing "
                "the confidence threshold. Messages use four labels: health "
                "questions, reminder setup, reminder replies and small talk. "
                "Symptoms remain questions even when mentioned in a dose reply.",
                "",
                "The health questions, danger-sign examples, adversarial messages "
                "and synthetic document fixtures were LLM drafted and reviewed "
                "row by row by a person. Team members recorded the voice set. "
                "Only the question set's tune half selects retrieval thresholds. "
                "No real patient data is used. Source citations remain in the "
                "corpus, danger-sign, lexicon and critical-value data files.",
                "",
                explanations(data, figures),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def publish(data: dict) -> None:
    report = Path("eval/report.md")
    health = report.read_text().split("\n## Model performance", 1)[0]
    health = health.replace(
        "Every number below is regenerated by that command; "
        "the per-item results are in `eval/results/`.",
        "Health results and per-item CSVs come from that command. Model performance "
        "and explainability are regenerated by "
        "`uv run python -m chhaaya.model_metrics` "
        "and published with `uv run python -m chhaaya.model_report`.",
    )
    report.write_text(health.rstrip() + "\n\n" + sections(data))
    readme = Path("README.md")
    text = readme.read_text()
    start, end = "<!-- model-evidence:start -->", "<!-- model-evidence:end -->"
    block = start + "\n" + sections(data, "eval/figures", compact=True) + end
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + block + after
    else:
        text = text.replace("[Full evaluation]", block + "\n\n[Full evaluation]", 1)
    readme.write_text(text)


def main() -> None:
    from chhaaya.model_metrics import load_report

    publish(load_report())
    print("Updated README and eval/report.md from saved model evidence.")


if __name__ == "__main__":
    main()
