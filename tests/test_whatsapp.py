import json

import httpx2

from chhaaya.whatsapp import WhatsAppClient


def fake_meta(requests: list[httpx2.Request]) -> httpx2.Client:
    """A stand-in for the Graph API."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        path = request.url.path
        if request.method == "POST" and path.endswith("/media"):
            return httpx2.Response(200, json={"id": "media-1"})
        if request.method == "POST" and path.endswith("/messages"):
            return httpx2.Response(200, json={"messages": [{"id": "wamid.out"}]})
        if request.url.host == "graph.facebook.com":
            return httpx2.Response(200, json={"url": "https://lookaside.example/m1"})
        return httpx2.Response(200, content=b"OggS-voice-note")

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_voice_reply_is_uploaded_then_sent_by_media_id():
    requests: list[httpx2.Request] = []
    client = WhatsAppClient("token", "123456", fake_meta(requests))

    media_id = client.upload_media(b"OggS-voice-note", "audio/ogg")
    wamid = client.send_audio("911111111111", media_id)

    upload, send = requests
    assert b"OggS-voice-note" in upload.read()
    assert json.loads(send.read()) == {
        "messaging_product": "whatsapp",
        "to": "911111111111",
        "type": "audio",
        "audio": {"id": "media-1", "voice": True},
    }
    assert wamid == "wamid.out"
    assert {r.headers["Authorization"] for r in requests} == {"Bearer token"}


def test_text_is_sent_and_inbound_media_is_downloaded_with_the_token():
    requests: list[httpx2.Request] = []
    client = WhatsAppClient("token", "123456", fake_meta(requests))

    assert client.send_text("911111111111", "namaste") == "wamid.out"
    assert client.download_media("media-9") == b"OggS-voice-note"

    # Meta's media URL refuses requests that don't carry the access token.
    assert requests[-1].url.host == "lookaside.example"
    assert requests[-1].headers["Authorization"] == "Bearer token"
