"""The worker: claims stored messages and turns them into replies (design section 3)."""

import functools
import logging
import re
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace

from sqlalchemy import Engine, create_engine, text

from chhaaya import danger_signs
from chhaaya.cases import Cases
from chhaaya.db import sqlalchemy_url
from chhaaya.document import DocumentAgent
from chhaaya.lexicon import Lexicon
from chhaaya.qa import QaAgent
from chhaaya.retrieval import KnowledgeBase, bge_m3
from chhaaya.sarvam import SarvamClient
from chhaaya.settings import Settings
from chhaaya.voice import send_voice
from chhaaya.whatsapp import WhatsAppClient

log = logging.getLogger("chhaaya.worker")
DEVANAGARI = re.compile("[\u0900-\u097f]")
# Sent when a voice note has no transcript in a supported language. One that cannot
# be transcribed at all is never screened for danger signs.
VOICE_UNREADABLE = (
    "We could not understand your voice note. Please type your message. "
    "If it is an emergency, call 108.\n\n"
    "Aapka voice note samajh nahi aaya. Kripya likhkar bhejein. "
    "Emergency ho to 108 par call karein."
)

# Take the oldest pending message. SKIP LOCKED makes a second worker pass over
# a message the first is claiming, so no message is processed twice.
CLAIM = text("""
    WITH claimed AS (
        UPDATE messages SET status = 'processing', claimed_at = now()
        WHERE id = (
            SELECT id FROM messages WHERE status = 'pending'
            ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1
        )
        RETURNING id, user_id, kind, body, media_id
    )
    SELECT c.id, c.user_id, c.kind, c.body, c.media_id,
           u.wa_id, u.role, u.asha_id, u.language, u.script
    FROM claimed c JOIN users u ON u.id = c.user_id
""")

# A message still processing this long after its claim lost its worker (killed, or
# the database went away before the result was saved). It is failed, not retried,
# so a reply that may already have gone out is never sent twice.
ABANDON = text("""
    UPDATE messages SET status = 'failed', error = 'worker stopped before finishing it'
    WHERE status = 'processing' AND claimed_at < now() - interval '10 minutes'
    RETURNING id
""")


@dataclass(frozen=True)
class Inbound:
    id: int
    user_id: int
    kind: str
    body: str | None
    media_id: str | None
    wa_id: str
    role: str
    asha_id: int | None
    language: str | None
    script: str | None


@dataclass
class Worker:
    engine: Engine
    whatsapp: WhatsAppClient
    sarvam: SarvamClient
    cases: Cases
    # "qa", "document" and "escalation" each take the message and reply.
    agents: dict[str, Callable[[Inbound], None]]

    def work_one(self) -> bool:
        """Process the oldest pending message; False when there is none."""
        with self.engine.begin() as conn:
            for abandoned in conn.execute(ABANDON).scalars():
                log.error("message %s was abandoned by a stopped worker", abandoned)
            row = conn.execute(CLAIM).one_or_none()
        if row is None:
            return False
        message = Inbound(**row._mapping)
        try:
            self.process(message)
        except Exception as error:
            log.exception("message %s failed", message.id)
            self.finish(message.id, "failed", f"{type(error).__name__}: {error}")
        else:
            self.finish(message.id, "done")
        return True

    def finish(self, message_id: int, status: str, error: str | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE messages SET status = :status, error = :error "
                    "WHERE id = :id"
                ),
                {"status": status, "error": error, "id": message_id},
            )

    def process(self, message: Inbound) -> None:
        if message.role == "asha":
            return self.route("escalation", message)
        understood = True
        if message.kind == "audio":
            message, understood = self.transcribe(message)
        if message.body and (signs := danger_signs.detect(message.body)):
            return self.escalate(message, signs)
        if not understood:
            self.whatsapp.send_text(message.wa_id, VOICE_UNREADABLE)
            raise ValueError("voice note is silent or in an unsupported language")
        if message.kind == "text":
            message = self.store_language(message)
        self.route("document" if message.kind == "image" else "qa", message)

    def route(self, agent: str, message: Inbound) -> None:
        if agent not in self.agents:
            raise LookupError(f"no {agent} agent is registered")
        self.agents[agent](message)

    def transcribe(self, message: Inbound) -> tuple[Inbound, bool]:
        """Put a voice note's transcript on the message, or tell the patient to type.

        False means the transcript is empty or in an unsupported language.
        """
        try:
            audio = self.whatsapp.download_media(message.media_id or "")
            body, language, script = self.sarvam.transcribe(audio, "audio/ogg")
        except Exception:
            self.whatsapp.send_text(message.wa_id, VOICE_UNREADABLE)
            raise
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE messages SET body = :body, language = :l WHERE id = :id"),
                {"body": body, "l": language, "id": message.id},
            )
        message = replace(message, body=body)
        if language is None or not body:
            return message, False
        return self.save_language(message, language, script), True

    def store_language(self, message: Inbound) -> Inbound:
        language, script = self.sarvam.identify_language(message.body or "")
        return self.save_language(message, language, script)

    def save_language(self, message: Inbound, language: str, script: str) -> Inbound:
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE users SET language = :l, script = :s WHERE id = :id"),
                {"l": language, "s": script, "id": message.user_id},
            )
        return replace(message, language=language, script=script)

    def escalate(self, message: Inbound, signs: list[str]) -> None:
        """Reply in text, open the ASHA's case even if that fails, then speak it."""
        log.warning("danger signs %s in message %s", signs, message.id)
        reply = urgent_reply(message)
        try:
            self.whatsapp.send_text(message.wa_id, reply)
        finally:
            if message.asha_id is not None:
                details = f"Danger signs: {', '.join(signs)}. Message: {message.body}"
                self.cases.open("danger_sign", message.user_id, details)
        send_voice(self.whatsapp, self.sarvam, message.wa_id, reply, message.language)
        if message.asha_id is None:
            raise LookupError("patient has no ASHA to alert")


def urgent_reply(message: Inbound) -> str:
    """The urgent reply in the patient's language. Reading it must not need Sarvam."""
    replies = danger_signs.URGENT_REPLY
    if message.language == "en":
        return replies["en"]
    if message.language == "hi":
        return replies["hi" if message.script == "Deva" else "hi-Latn"]
    if DEVANAGARI.search(message.body or ""):
        return replies["hi"]
    # Latin script from someone we know nothing about: English or Hinglish.
    return f"{replies['en']}\n\n{replies['hi-Latn']}"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    engine = create_engine(sqlalchemy_url(settings.database_url))
    whatsapp = WhatsAppClient(
        settings.whatsapp_access_token, settings.whatsapp_phone_number_id
    )
    sarvam = SarvamClient(settings.sarvam_api_key)
    # BGE-M3 loads on the first answer to a question, so a danger sign is never
    # held up by the 2 GB model download.
    load = functools.cache(bge_m3)
    cases = Cases(engine, whatsapp, sarvam, lambda texts: load()(texts))
    worker = Worker(engine, whatsapp, sarvam, cases, {"escalation": cases.answer})
    lexicon = functools.cache(Lexicon.load)
    worker.agents["document"] = DocumentAgent(sarvam, whatsapp, cases, lexicon).read
    kb = KnowledgeBase(engine, lambda texts: load()(texts))
    worker.agents["qa"] = QaAgent(kb, sarvam, whatsapp, cases, worker.escalate).answer
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    log.info("worker started")
    while not stop.is_set():
        if not worker.work_one():
            stop.wait(1)
    log.info("worker stopped")


if __name__ == "__main__":
    main()
