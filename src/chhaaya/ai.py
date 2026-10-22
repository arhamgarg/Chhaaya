"""Select Claude or the manual Sarvam fallback, with speech enabled separately."""

from typing import Protocol

import httpx2

from chhaaya.claude import ClaudeClient
from chhaaya.sarvam import SarvamClient


class ProviderSettings(Protocol):
    """The settings AIClient reads: the service's and the browser demo's."""

    @property
    def ai_provider(self) -> str: ...
    @property
    def anthropic_api_key(self) -> str: ...
    @property
    def sarvam_api_key(self) -> str: ...
    @property
    def speech_enabled(self) -> bool: ...


class AIClient:
    def __init__(
        self, settings: ProviderSettings, http: httpx2.Client | None = None
    ) -> None:
        sarvam = SarvamClient(settings.sarvam_api_key, http)
        self.text = (
            ClaudeClient(settings.anthropic_api_key, http)
            if settings.ai_provider == "claude"
            else sarvam
        )
        self.speech_enabled = settings.speech_enabled
        self.speech = sarvam if self.speech_enabled else None

    def chat(self, system: str, user: str) -> str:
        return self.text.chat(system, user)

    def identify_language(self, text: str) -> tuple[str, str]:
        return self.text.identify_language(text)

    def read_document(self, image: bytes) -> str:
        return self.text.read_document(image)

    def transcribe(self, audio: bytes, mime_type: str) -> tuple[str, str | None, str]:
        if self.speech is None:
            raise ValueError("speech is disabled; send a typed message")
        return self.speech.transcribe(audio, mime_type)

    def synthesize(self, text: str, language: str) -> bytes:
        if self.speech is None:
            raise ValueError("speech is disabled")
        return self.speech.synthesize(text, language)
