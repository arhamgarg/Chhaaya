"""Client risks: leaked secrets, reasoning as an answer, truncated/refused text,
invalid language JSON, and altered document numbers at the provider boundary.
"""

import base64
import json
import logging
from pathlib import Path

import httpx2
import pytest

from chhaaya.claude import ClaudeClient


def response(*parts, stop="end_turn"):
    return {"stop_reason": stop, "content": list(parts)}


def client(handler):
    return ClaudeClient(
        "synthetic-secret", httpx2.Client(transport=httpx2.MockTransport(handler))
    )


def test_chat_separates_instructions_and_returns_only_completed_answer_text():
    def handler(request):
        assert request.url == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "synthetic-secret"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.read())
        assert body["system"] == "Only cite supplied sources."
        assert body["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "Give a dose."}]}
        ]
        return httpx2.Response(
            200,
            json=response(
                {"type": "thinking", "thinking": "private reasoning"},
                {"type": "text", "text": "Ask your ASHA "},
                {"type": "text", "text": "about medicines [12]."},
            ),
        )

    assert client(handler).chat("Only cite supplied sources.", "Give a dose.") == (
        "Ask your ASHA about medicines [12]."
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        response({"type": "text", "text": " "}),
        response({"type": "thinking", "thinking": "internal only"}),
        response({"type": "text", "text": "partial"}, stop="max_tokens"),
        response({"type": "text", "text": "refused"}, stop="refusal"),
        response({"type": "tool_use", "name": "unrequested"}, stop="tool_use"),
    ],
)
def test_unusable_answers_fail_without_returning_partial_or_reasoning_text(payload):
    with pytest.raises(ValueError, match="Claude"):
        client(lambda _: httpx2.Response(200, json=payload)).chat("system", "user")


def test_photo_is_read_inline_without_an_upload_job():
    photo = (Path(__file__).parents[1] / "eval/sets/documents/rx-01.jpg").read_bytes()

    def handler(request):
        body = json.loads(request.read())
        assert body["messages"][0]["content"][0] == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(photo).decode(),
            },
        }
        return httpx2.Response(
            200, json=response({"type": "text", "text": "Haemoglobin: 6.8 g/dL"})
        )

    assert client(handler).read_document(photo) == "Haemoglobin: 6.8 g/dL"


@pytest.mark.parametrize(
    ("language", "expected"),
    [("hi-IN", "hi"), ("en-US", "en"), ("EN-gb", "en")],
)
def test_language_json_is_requested_as_a_schema_and_normalized(language, expected):
    def handler(request):
        body = json.loads(request.read())
        assert body["output_config"]["format"]["type"] == "json_schema"
        text = json.dumps({"language_code": language, "script_code": "Latn"})
        return httpx2.Response(200, json=response({"type": "text", "text": text}))

    assert client(handler).identify_language("hello") == (expected, "Latn")


@pytest.mark.parametrize(
    "text",
    ["not json", '{"language_code":"ta-IN","script_code":"Latn"}'],
)
def test_invalid_or_unsupported_language_fails_instead_of_guessing(text):
    with pytest.raises(ValueError):
        client(
            lambda _: httpx2.Response(
                200, json=response({"type": "text", "text": text})
            )
        ).identify_language("hello")


def test_http_errors_log_status_and_request_id_without_credentials_or_patient_text(
    caplog,
):
    with caplog.at_level(logging.ERROR), pytest.raises(httpx2.HTTPStatusError):
        client(
            lambda _: httpx2.Response(
                429,
                json={"error": {"message": "quota"}},
                headers={"request-id": "req-123"},
            )
        ).chat("private prompt", "private patient text")
    assert "req-123" in caplog.text and "429" in caplog.text
    assert "synthetic-secret" not in caplog.text
    assert "private patient text" not in caplog.text
