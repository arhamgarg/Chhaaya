import base64
import logging

import httpx2
import pytest

from chhaaya.sarvam import SarvamClient


def client(handler) -> SarvamClient:
    return SarvamClient("key", httpx2.Client(transport=httpx2.MockTransport(handler)))


def test_a_voice_note_is_sent_as_audio_and_comes_back_as_text_with_its_language():
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            200, json={"transcript": "मुझे बुखार है", "language_code": "hi-IN"}
        )

    text, language, script = client(handler).transcribe(b"OggS-voice", "audio/ogg")

    assert (text, language, script) == ("मुझे बुखार है", "hi", "Deva")
    body = seen[0].read()
    assert b"OggS-voice" in body
    assert b"saaras:v4" in body
    assert b'name="mode"' not in body  # mode applies only to saaras:v3
    assert seen[0].headers["api-subscription-key"] == "key"


def test_hinglish_speech_comes_back_in_the_latin_script():
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json={"transcript": "mujhe bukhar hai", "language_code": "hi-IN"}
        )

    assert client(handler).transcribe(b"x", "audio/ogg")[1:] == ("hi", "Latn")


def test_a_reply_is_synthesised_in_the_patients_language_and_decoded():
    wav = b"RIFF....WAVEfmt "

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert b'"hi-IN"' in request.read()
        assert b"bulbul:v4-flash" in request.read()
        return httpx2.Response(200, json={"audios": [base64.b64encode(wav).decode()]})

    assert client(handler).synthesize("आपको आराम करना चाहिए", "hi") == wav


def test_a_failed_call_is_logged_with_its_request_id_and_still_raises(
    caplog: pytest.LogCaptureFixture,
):
    def handler(_: httpx2.Request) -> httpx2.Response:
        error = {"request_id": "req-123", "message": "busy", "code": "rate_limit"}
        return httpx2.Response(429, json={"error": error})

    with caplog.at_level(logging.ERROR), pytest.raises(httpx2.HTTPStatusError):
        client(handler).synthesize("hello", "en")

    assert "req-123" in caplog.text
    assert "text-to-speech" in caplog.text


@pytest.mark.parametrize("language_code", ["ta-IN", None])
def test_speech_in_an_unsupported_language_keeps_its_transcript(language_code):
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json={"transcript": "vanakkam", "language_code": language_code}
        )

    text, language, _ = client(handler).transcribe(b"x", "audio/ogg")

    assert (text, language) == ("vanakkam", None)
