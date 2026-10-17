import base64
import logging

import httpx2
import pytest
from test_audio import tone

from chhaaya.sarvam import SarvamClient
from chhaaya.voice import send_reply
from chhaaya.whatsapp import WhatsAppClient


def clients(speech_works: bool):
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if request.url.host == "api.sarvam.ai":
            if not speech_works:
                return httpx2.Response(500, json={"error": {"request_id": "r1"}})
            audio = base64.b64encode(tone(0.5)).decode()
            return httpx2.Response(200, json={"audios": [audio]})
        if request.url.path.endswith("/media"):
            return httpx2.Response(200, json={"id": "media-up"})
        return httpx2.Response(200, json={"messages": [{"id": "wamid.out"}]})

    http = httpx2.Client(transport=httpx2.MockTransport(handler))
    return WhatsAppClient("token", "123", http), SarvamClient("key", http), requests


def test_a_reply_goes_out_as_text_then_as_an_opus_voice_note():
    whatsapp, sarvam, requests = clients(speech_works=True)

    send_reply(whatsapp, sarvam, "911111111111", "Aap theek ho jaayenge", "hi")

    paths = [request.url.path for request in requests]
    assert paths == [
        "/v23.0/123/messages",
        "/text-to-speech",
        "/v23.0/123/media",
        "/v23.0/123/messages",
    ]
    upload, voice = requests[2].read(), requests[3].read()
    assert b"OggS" in upload and b"audio/ogg; codecs=opus" in upload
    assert b'"media-up"' in voice


def test_when_speech_fails_the_text_has_already_gone_and_nothing_raises(
    caplog: pytest.LogCaptureFixture,
):
    whatsapp, sarvam, requests = clients(speech_works=False)

    with caplog.at_level(logging.ERROR):
        send_reply(whatsapp, sarvam, "911111111111", "Aap theek ho jaayenge", "hi")

    assert [r.url.path for r in requests] == ["/v23.0/123/messages", "/text-to-speech"]
    assert "r1" in caplog.text
    assert "911111111111" not in caplog.text


def test_without_a_known_language_only_text_is_sent(caplog: pytest.LogCaptureFixture):
    whatsapp, sarvam, requests = clients(speech_works=True)

    with caplog.at_level(logging.ERROR):
        send_reply(whatsapp, sarvam, "911111111111", "Namaste", None)

    assert [r.url.path for r in requests] == ["/v23.0/123/messages"]
    assert caplog.records == []
