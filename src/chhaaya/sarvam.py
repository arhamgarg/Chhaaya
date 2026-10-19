"""The Sarvam client: the only code that talks to api.sarvam.ai."""

import base64
import io
import logging
import re
import time
import zipfile

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
# Sarvam Vision reads a photo in a job that is polled. Hindi also reads the
# English that most Indian lab reports and prescriptions are printed in.
VISION_LANGUAGE = "hi-IN"
VISION_POLL = 2  # seconds between status checks
VISION_TIMEOUT = 120  # seconds before a job still running is given up


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

    def read_document(self, image: bytes) -> str:
        """The Markdown Sarvam Vision reads off a JPEG photo of a document."""
        job = self._post(
            "doc-ai/v1/job/digitise",
            data={"language": VISION_LANGUAGE, "output_format": "md"},
            files={"file": ("photo.jpg", image, "image/jpeg")},
        ).json()["job_id"]
        deadline = time.monotonic() + VISION_TIMEOUT
        while (status := self._get(f"doc-ai/v1/job/{job}/status")["status"]) in (
            "pending",
            "running",
        ):
            if time.monotonic() > deadline:
                raise TimeoutError(f"sarvam vision job {job} still {status}")
            time.sleep(VISION_POLL)
        # One photo is one page, so partially_completed means nothing was read.
        if status != "completed":
            raise ValueError(f"sarvam vision job {job} {status}")
        url = self._get(f"doc-ai/v1/job/{job}/download-url")["url"]
        # A signed URL on Sarvam's storage: the API key is not sent there.
        download = self.http.get(url)
        download.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(download.content)) as zf:
            [markdown] = (n for n in zf.namelist() if n.endswith(".md"))
            return zf.read(markdown).decode("utf-8")

    def _get(self, endpoint: str) -> dict:
        return self._request("GET", endpoint).json()

    def _post(self, endpoint: str, **request) -> httpx2.Response:
        return self._request("POST", endpoint, **request)

    def _request(self, method: str, endpoint: str, **request) -> httpx2.Response:
        response = self.http.request(
            method, f"{API_URL}/{endpoint}", headers=self.headers, **request
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
