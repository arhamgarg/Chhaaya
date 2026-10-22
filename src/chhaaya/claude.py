"""Anthropic Messages API for text, language identification and document OCR."""

import base64
import logging

import httpx2
from pydantic import BaseModel

from chhaaya.gemini import (
    LANGUAGE_PROMPT,
    OCR_PROMPT,
    Language,
    image_media_type,
    supported,
)

log = logging.getLogger("chhaaya.claude")
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-5-5"
TIMEOUT = 120


class ClaudeClient:
    def __init__(self, api_key: str, http: httpx2.Client | None = None) -> None:
        self.http = http or httpx2.Client(timeout=TIMEOUT)
        self.headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

    def _text(
        self, system: str, content: list[dict], schema: type[BaseModel] | None = None
    ) -> str:
        output: dict = {}
        if schema is not None:
            output["format"] = {
                "type": "json_schema",
                "schema": schema.model_json_schema(),
            }
        response = self.http.post(
            API_URL,
            headers=self.headers,
            json={
                "model": MODEL,
                "max_tokens": 4096,
                # This model uses between_tools to disable up-front thinking.
                "thinking": {"type": "between_tools"},
                "system": system,
                "messages": [{"role": "user", "content": content}],
                "output_config": output,
            },
        )
        if response.is_error:
            log.error(
                "Claude %s HTTP %s, request id %s",
                MODEL,
                response.status_code,
                response.headers.get("request-id", "unavailable"),
            )
        response.raise_for_status()
        found = response.json()
        if found.get("stop_reason") != "end_turn":
            raise ValueError(f"Claude generation stopped: {found.get('stop_reason')}")
        text = "".join(
            part["text"]
            for part in found.get("content", [])
            if part.get("type") == "text"
        )
        if not text.strip():
            raise ValueError("Claude returned empty answer text")
        return text

    def chat(self, system: str, user: str) -> str:
        return self._text(system, [{"type": "text", "text": user}])

    def identify_language(self, text: str) -> tuple[str, str]:
        found = Language.model_validate_json(
            self._text(
                LANGUAGE_PROMPT, [{"type": "text", "text": text[:1000]}], Language
            )
        )
        language = supported(found.language_code, found.script_code)
        assert found.script_code is not None  # supported rejects an unknown script
        return language, found.script_code

    def read_document(self, image: bytes) -> str:
        return self._text(
            OCR_PROMPT,
            [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type(image),
                        "data": base64.b64encode(image).decode(),
                    },
                }
            ],
        )
