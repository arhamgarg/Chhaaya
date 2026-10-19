"""The WhatsApp Cloud API client: the only code that talks to graph.facebook.com."""

import httpx2

GRAPH_URL = "https://graph.facebook.com/v23.0"


class WhatsAppClient:
    def __init__(
        self, access_token: str, phone_number_id: str, http: httpx2.Client | None = None
    ) -> None:
        self.phone_number_id = phone_number_id
        self.http = http or httpx2.Client(timeout=10)
        self.headers = {"Authorization": f"Bearer {access_token}"}

    def send_text(self, to: str, body: str) -> str:
        return self._send(to, {"type": "text", "text": {"body": body}})

    def send_audio(self, to: str, media_id: str) -> str:
        return self._send(
            to, {"type": "audio", "audio": {"id": media_id, "voice": True}}
        )

    def send_image(self, to: str, media_id: str, caption: str) -> str:
        return self._send(
            to, {"type": "image", "image": {"id": media_id, "caption": caption}}
        )

    def send_template(
        self, to: str, name: str, language: str, params: list[str]
    ) -> str:
        """Send an approved template, the only kind allowed outside the 24h window."""
        parameters = [{"type": "text", "text": p} for p in params]
        template = {
            "name": name,
            "language": {"code": language},
            "components": [{"type": "body", "parameters": parameters}],
        }
        return self._send(to, {"type": "template", "template": template})

    def upload_media(self, data: bytes, mime_type: str) -> str:
        response = self.http.post(
            f"{GRAPH_URL}/{self.phone_number_id}/media",
            headers=self.headers,
            data={"messaging_product": "whatsapp", "type": mime_type},
            files={"file": ("media", data, mime_type)},
        )
        response.raise_for_status()
        return response.json()["id"]

    def download_media(self, media_id: str) -> bytes:
        lookup = self.http.get(f"{GRAPH_URL}/{media_id}", headers=self.headers)
        lookup.raise_for_status()
        # The media URL needs the access token too.
        response = self.http.get(lookup.json()["url"], headers=self.headers)
        response.raise_for_status()
        return response.content

    def _send(self, to: str, message: dict) -> str:
        response = self.http.post(
            f"{GRAPH_URL}/{self.phone_number_id}/messages",
            headers=self.headers,
            json={"messaging_product": "whatsapp", "to": to, **message},
        )
        response.raise_for_status()
        return response.json()["messages"][0]["id"]
