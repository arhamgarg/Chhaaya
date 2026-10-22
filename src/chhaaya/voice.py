"""Replies: always text, plus the same words as a voice note when that works."""

import logging

from chhaaya.ai import AIClient
from chhaaya.audio import to_ogg_opus
from chhaaya.whatsapp import WhatsAppClient

log = logging.getLogger("chhaaya.voice")


def send_reply(
    whatsapp: WhatsAppClient,
    ai: AIClient,
    to: str,
    text: str,
    language: str | None,
) -> None:
    """Send text first so a patient is never left in silence, then a voice note."""
    whatsapp.send_text(to, text)
    send_voice(whatsapp, ai, to, text, language)


def send_voice(
    whatsapp: WhatsAppClient,
    ai: AIClient,
    to: str,
    text: str,
    language: str | None,
) -> None:
    """Send text that already went out as a voice note too, if the language is known.

    A failure is logged without the phone number and goes no further: the text
    reply has already been sent.
    """
    if language is None or not ai.speech_enabled:
        return
    try:
        ogg = to_ogg_opus(ai.synthesize(text, language))
        media_id = whatsapp.upload_media(ogg, "audio/ogg; codecs=opus")
        whatsapp.send_audio(to, media_id)
    except Exception:
        log.exception("voice reply failed; the text reply was sent")
