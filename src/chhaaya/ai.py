"""Select Claude or the manual Sarvam fallback, with speech enabled separately."""

import httpx2

from chhaaya.claude import ClaudeClient
from chhaaya.sarvam import SarvamClient
from chhaaya.settings import Settings


class AIClient:
    def __init__(self, settings: Settings, http: httpx2.Client | None = None) -> None:
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
