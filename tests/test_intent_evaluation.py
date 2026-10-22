"""Measure saved models and persist results, including a partial provider failure.

Risks: wrong macro-F1, language mixing, missing passes, cached classifications,
silently accepted API failures, and paid successes lost after a later failure.
"""

import csv
import json

import httpx2
import pytest
from test_ai import settings
from test_intent_flow import saved_classifier

from chhaaya import intent, sarvam
from chhaaya import settings as configuration
from chhaaya.intent import evaluate


@pytest.mark.parametrize("fail_second_request", [False, True])
def test_benchmark_scores_cpu_predictions_and_keeps_completed_paid_requests(
    tmp_path, monkeypatch, fail_second_request
):
    checkpoint = tmp_path / "model"
    saved_classifier(checkpoint, "question", 8.0)
    (checkpoint / "training.json").write_text('{"fixture":true}')
    data = tmp_path / "data/intent"
    data.mkdir(parents=True)
    samples = {
        "en": ["How does ORS work?", "remind me tonight", "taken", "hello"],
        "hi": ["ओआरएस क्या है?", "रात में याद दिलाना", "ले ली", "नमस्ते"],
        "hinglish": ["ORS kya hai?", "raat ko yaad dilana", "le li", "shukriya"],
    }
    with (data / "test.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "language"])
        for language, texts in samples.items():
            for label, message in zip(
                ["question", "reminder_setup", "reminder_reply", "smalltalk"],
                texts,
                strict=True,
            ):
                writer.writerow([message, label, language])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        configuration,
        "Settings",
        lambda: settings(anthropic_api_key="synthetic", sarvam_api_key="synthetic"),
    )
    requests = []
    clock = [0.0]
    previous_request = [-2.0]

    def now():
        clock[0] += 0.001
        return clock[0]

    def wait(seconds):
        clock[0] += seconds

    monkeypatch.setattr(intent, "perf_counter", now)
    monkeypatch.setattr(intent, "sleep", wait, raising=False)

    def boundary(request):
        instant = now()
        if instant - previous_request[0] < 2.0:
            return httpx2.Response(429, json={"error": "synthetic rate limit"})
        previous_request[0] = instant
        requests.append(request)
        if fail_second_request and len(requests) == 2:
            return httpx2.Response(503, json={"error": "synthetic provider down"})
        response_format = json.loads(request.read()).get("response_format", {})
        constrained = response_format.get("json_schema", {})
        schema = constrained.get("schema", {})
        structured = (
            response_format.get("type") == "json_schema"
            and constrained.get("strict") is True
            and schema.get("additionalProperties") is False
            and schema.get("required") == ["intent"]
            and set(schema.get("properties", {}).get("intent", {}).get("enum", []))
            == {"question", "reminder_setup", "reminder_reply", "smalltalk"}
        )
        content = '{"intent":"question"}'
        if not structured:
            content = f"```json\n{content}\n```"
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": content},
                    }
                ]
            },
        )

    real_client = sarvam.SarvamClient
    monkeypatch.setattr(
        sarvam,
        "SarvamClient",
        lambda key: real_client(
            key, httpx2.Client(transport=httpx2.MockTransport(boundary))
        ),
    )
    output = tmp_path / "results/intent.json"
    if fail_second_request:
        with pytest.raises(httpx2.HTTPStatusError, match="503"):
            evaluate(checkpoint, output, 3)
        assert not output.exists()
    else:
        evaluate(checkpoint, output, 3)
        report = json.loads(output.read_text())
        assert report["runs"] == 3
        assert len(report["results"]) == 9
        for result in report["results"]:
            assert result["n"] == 4
            assert result["macro_f1"] == (1.0 if result["model"] == "keywords" else 0.1)
            assert len(result["run_medians_ms"]) == 3
            assert result["median_ms"] > 0
    rows = list(csv.DictReader(output.with_suffix(".csv").open()))
    assert len(rows) == (73 if fail_second_request else 108)
    assert len([r for r in rows if r["model"] == "sarvam-105b"]) == (
        1 if fail_second_request else 36
    )
    assert all(r["predicted"] == "question" for r in rows if r["model"] == "muril")
