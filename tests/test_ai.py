"""Provider selection must never silently switch a failed health answer. Speech
must stay off without explicit opt-in, and credentials must match the selection.
"""

import json

import httpx2
import pytest
from pydantic import ValidationError

from chhaaya.ai import AIClient
from chhaaya.settings import Settings


def settings(**overrides):
    return Settings(
        _env_file=None,
        postgres_password="unused",
        database_url="postgresql://unused",
        whatsapp_verify_token="unused",
        whatsapp_app_secret="unused",
        whatsapp_access_token="unused",
        whatsapp_phone_number_id="unused",
        **overrides,
    )


@pytest.mark.parametrize("provider", ["claude", "gemini"])
def test_selected_provider_answers_without_calling_the_other_provider(provider):
    def handler(request):
        if provider == "claude":
            assert request.url.host == "api.anthropic.com"
            payload = {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Ask your ASHA [12]."}],
            }
        else:
            assert request.url.host == "generativelanguage.googleapis.com"
            payload = {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "Ask your ASHA [12]."}]},
                    }
                ]
            }
        return httpx2.Response(200, json=payload)

    ai = AIClient(
        settings(
            ai_provider=provider,
            anthropic_api_key="claude" if provider == "claude" else "",
            gemini_api_key="gemini" if provider == "gemini" else "",
        ),
        httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    assert ai.chat("Only cite sources", "Question") == "Ask your ASHA [12]."


def test_a_claude_failure_is_reported_without_sending_patient_text_to_gemini():
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx2.Response(429, json={"error": {"message": "quota"}})

    ai = AIClient(
        settings(anthropic_api_key="claude", gemini_api_key="gemini"),
        httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    with pytest.raises(httpx2.HTTPStatusError):
        ai.chat("system", "patient text")
    assert calls == ["api.anthropic.com"]


def test_deferred_speech_does_not_call_a_paid_provider():
    def handler(request):
        pytest.fail(f"speech must not call {request.url.host}")

    ai = AIClient(
        settings(anthropic_api_key="claude", gemini_api_key="gemini"),
        httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    with pytest.raises(ValueError, match="speech is disabled"):
        ai.transcribe(b"OggS-synthetic", "audio/ogg")
    with pytest.raises(ValueError, match="speech is disabled"):
        ai.synthesize("hello", "en")


def test_opted_in_speech_keeps_gemini_and_preserves_an_unsupported_transcript():
    def handler(request):
        assert request.url.host == "generativelanguage.googleapis.com"
        text = json.dumps(
            {
                "transcript": "baby has convulsions",
                "language_code": "ta-IN",
                "script_code": "Latn",
            }
        )
        return httpx2.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": text}]},
                    }
                ]
            },
        )

    ai = AIClient(
        settings(
            anthropic_api_key="claude", gemini_api_key="gemini", speech_enabled=True
        ),
        httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    assert ai.transcribe(b"OggS-synthetic", "audio/ogg") == (
        "baby has convulsions",
        None,
        "Latn",
    )


@pytest.mark.parametrize(
    "configuration",
    [
        {"ai_provider": "claude", "anthropic_api_key": ""},
        {"ai_provider": "gemini", "gemini_api_key": ""},
        {"anthropic_api_key": "claude", "speech_enabled": True, "gemini_api_key": ""},
        {"ai_provider": "unknown"},
    ],
)
def test_configuration_rejects_missing_credentials_before_processing(configuration):
    with pytest.raises(ValidationError):
        settings(**configuration)
