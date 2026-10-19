"""The Sarvam client: the only code that talks to api.sarvam.ai."""

import base64
import logging
import re

import httpx2

log = logging.getLogger("chhaaya.sarvam")

API_URL = "https://api.sarvam.ai"
LANGUAGES = {"hi-IN": "hi", "en-IN": "en"}
SPEECH_CODES = {language: code for code, language in LANGUAGES.items()}
STT_MODEL = "saaras:v4"
TTS_MODEL = "bulbul:v4-flash"
CHAT_MODEL = "sarvam-105b"
TTS_MAX_CHARACTERS = 2500
DEVANAGARI = re.compile("[ऀ-ॿ]")
SPEECH_TIMEOUT = 30  # seconds; speech calls are slower than text calls
CHAT_TIMEOUT = 60  # seconds; Sarvam-105B reasons before it answers


def supported(language_code: str | None, script_code: str | None) -> str:
    """The language ("hi" or "en") for a Sarvam code, or ValueError."""
    language = LANGUAGES.get(language_code or "")
    if language is None or script_code not in (None, "Deva", "Latn"):
        raise ValueError(
            f"unsupported language {language_code} in script {script_code}"
        )
    return language


class SarvamClient:
    def __init__(self, api_key: str, http: httpx2.Client | None = None) -> None:
        self.http = http or httpx2.Client(timeout=10)
        self.headers = {"api-subscription-key": api_key}

    def identify_language(self, text: str) -> tuple[str, str]:
        """The language ("hi" or "en") and script ("Deva" or "Latn") of typed text."""
        found = self._post("text-lid", json={"input": text[:1000]}).json()
        script = found["script_code"]
        return supported(found["language_code"], script), script

    def transcribe(self, audio: bytes, mime_type: str) -> tuple[str, str | None, str]:
        """Speech to text: the transcript, its language (None if unsupported), script.

        An unsupported language still returns its transcript, so the caller can
        screen it for danger signs.
        """
        found = self._post(
            "speech-to-text",
            data={"model": STT_MODEL, "language_code": "unknown"},
            files={"file": ("voice", audio, mime_type)},
            timeout=SPEECH_TIMEOUT,
        ).json()
        language = LANGUAGES.get(found["language_code"] or "")
        script = "Deva" if DEVANAGARI.search(found["transcript"]) else "Latn"
        return found["transcript"], language, script

    def synthesize(self, text: str, language: str) -> bytes:
        """Text to speech: WAV audio in the given language ("hi" or "en")."""
        if len(text) > TTS_MAX_CHARACTERS:
            raise ValueError(f"{len(text)} characters is too long to speak")
        found = self._post(
            "text-to-speech",
            json={
                "text": text,
                "language_code": SPEECH_CODES[language],
                "model": TTS_MODEL,
            },
            timeout=SPEECH_TIMEOUT,
        ).json()
        return base64.b64decode("".join(found["audios"]))

    def chat(self, system: str, user: str) -> str:
        """Sarvam-105B's reply to one user message under a system prompt."""
        found = self._post(
            "v1/chat/completions",
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=CHAT_TIMEOUT,
        ).json()
        choice = found["choices"][0]
        # Reasoning counts against max_tokens, so a long one can leave no answer.
        if choice["finish_reason"] != "stop" or not choice["message"]["content"]:
            raise ValueError(f"sarvam chat stopped early: {choice['finish_reason']}")
        return choice["message"]["content"]

    def _post(self, endpoint: str, **request) -> httpx2.Response:
        response = self.http.post(
            f"{API_URL}/{endpoint}", headers=self.headers, **request
        )
        if response.is_error:
            try:
                request_id = response.json()["error"]["request_id"]
            except (ValueError, KeyError, TypeError):
                request_id = None
            log.error(
                "sarvam %s failed with %s, request id %s",
                endpoint,
                response.status_code,
                request_id,
            )
        response.raise_for_status()
        return response
