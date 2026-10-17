"""Escalation cases: alert the ASHA, relay her answer (design section 5.4)."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import Connection, Engine, insert, text

from chhaaya.db import KbChunk
from chhaaya.sarvam import SarvamClient
from chhaaya.voice import send_reply
from chhaaya.whatsapp import WhatsAppClient

if TYPE_CHECKING:
    from chhaaya.retrieval import Embed
    from chhaaya.worker import Inbound

# Answers to these become knowledge-base chunks; the rest are about one patient.
QUESTION_CAUSES = {"weak_evidence", "citation_failed"}
LABELS = {
    "danger_sign": "danger sign",
    "weak_evidence": "question",
    "citation_failed": "question",
    "critical_lab_value": "critical lab value",
    "prescription_confirmation": "prescription to confirm",
    "missed_doses": "missed doses",
}
ANSWER = re.compile(r"\s*(\d+)\s+(\S.*)", re.DOTALL)
USAGE = (
    "To answer a case, start with its number, for example: "
    "12 Give ORS and bring the child to the PHC today."
)
SHARED_WARNING = (
    "\nYour answer will also be saved to help other patients, so leave out "
    "names and personal details."
)
NOT_OPEN = "Case {number} is not open for you. Check the number and try again."
NOTICE = {
    "en": "We have sent your question to your ASHA. She will reply here.",
    "hi": "हमने आपका सवाल आपकी आशा दीदी को भेज दिया है। वे यहीं जवाब देंगी।",
    "hi-Latn": "Humne aapka sawal aapki ASHA didi ko bhej diya hai. Woh yahin "
    "jawab dengi.",
}
REPLY_PREFIX = {
    "en": "Your ASHA replied:",
    "hi": "आपकी आशा दीदी का जवाब:",
    "hi-Latn": "Aapki ASHA didi ka jawab:",
}
TEMPLATE_LANGUAGES = {"hi": "hi", "en": "en"}
# The alert quotes at most this many characters of the details, so it stays within
# the 1024-character body of the asha_case_alert template. The case keeps them all.
SUMMARY_LIMIT = 500


def in_language(texts: dict[str, str], language: str | None, script: str | None) -> str:
    """The text for a user's language; English and Hinglish when it is unknown."""
    if language == "en":
        return texts["en"]
    if language == "hi":
        return texts["hi" if script == "Deva" else "hi-Latn"]
    return f"{texts['en']}\n\n{texts['hi-Latn']}"


def one_line(value: str) -> str:
    """Template parameters may not contain newlines or runs of spaces."""
    return " ".join(value.split())


def summary(details: str) -> str:
    """The case details, cut to SUMMARY_LIMIT characters for the alert."""
    if len(details) <= SUMMARY_LIMIT:
        return details
    return details[: SUMMARY_LIMIT - 1] + "…"


def in_window(conn: Connection, user_id: int) -> bool:
    """Whether the user wrote in the last 24 hours, so free-form text is allowed."""
    return conn.execute(
        text(
            "SELECT EXISTS (SELECT 1 FROM messages WHERE user_id = :u "
            "AND direction = 'inbound' AND created_at > now() - interval '24 hours')"
        ),
        {"u": user_id},
    ).scalar_one()


@dataclass
class Cases:
    engine: Engine
    whatsapp: WhatsAppClient
    sarvam: SarvamClient
    embed: "Embed"

    def open(self, cause: str, patient_id: int, details: str) -> int:
        """Open a numbered case, alert the patient's ASHA and tell the patient.

        A danger-sign patient has already had the urgent reply, so is not told
        again, and neither is a patient outside the 24-hour window. The case is
        saved before any message, so a failed alert still leaves it open for the
        ASHA.
        """
        with self.engine.begin() as conn:
            people = conn.execute(
                text(
                    "SELECT p.wa_id, p.language, p.script, p.asha_id, "
                    "a.wa_id AS asha_wa_id, a.language AS asha_language "
                    "FROM users p LEFT JOIN users a ON a.id = p.asha_id "
                    "WHERE p.id = :id"
                ),
                {"id": patient_id},
            ).one()
            if people.asha_id is None:
                raise LookupError("patient has no ASHA to alert")
            number = conn.execute(
                text(
                    "INSERT INTO cases (cause, patient_id, asha_id, details) "
                    "VALUES (:cause, :patient, :asha, :details) RETURNING number"
                ),
                {
                    "cause": cause,
                    "patient": patient_id,
                    "asha": people.asha_id,
                    "details": details,
                },
            ).scalar_one()
            asha_in_window = in_window(conn, people.asha_id)
            patient_in_window = in_window(conn, patient_id)
        label = LABELS[cause]
        shared = SHARED_WARNING if cause in QUESTION_CAUSES else ""
        if asha_in_window:
            self.whatsapp.send_text(
                people.asha_wa_id,
                f"Case {number}: {label} from {people.wa_id}.\n{summary(details)}\n\n"
                f"Reply with: {number} <your answer>{shared}",
            )
        else:
            self.whatsapp.send_template(
                people.asha_wa_id,
                "asha_case_alert",
                TEMPLATE_LANGUAGES.get(people.asha_language, "hi"),
                [str(number), label, people.wa_id, one_line(summary(details))],
            )
        # No approved template carries the notice, so outside the window the
        # patient hears from the case only when the ASHA's case_update arrives.
        if cause != "danger_sign" and patient_in_window:
            notice = in_language(NOTICE, people.language, people.script)
            send_reply(
                self.whatsapp, self.sarvam, people.wa_id, notice, people.language
            )
        return number

    def answer(self, message: "Inbound") -> None:
        """The escalation agent: an ASHA's "12 <answer>" closes case 12."""
        match = ANSWER.fullmatch(message.body or "")
        if match is None:
            self.whatsapp.send_text(message.wa_id, USAGE)
            return
        number, reply = int(match[1]), match[2].strip()
        # The case is closed only once the patient has the answer: if relaying
        # fails, everything rolls back and the case stays open.
        with self.engine.begin() as conn:
            case = conn.execute(
                text(
                    "SELECT c.cause, c.details, p.id AS patient_id, p.wa_id, "
                    "p.language, p.script FROM cases c "
                    "JOIN users p ON p.id = c.patient_id "
                    "WHERE c.number = :n AND c.asha_id = :asha AND c.status = 'open' "
                    "FOR UPDATE OF c"
                ),
                {"n": number, "asha": message.user_id},
            ).one_or_none()
            if case is None:
                self.whatsapp.send_text(message.wa_id, NOT_OPEN.format(number=number))
                return
            if case.cause in QUESTION_CAUSES:
                # Later patients may be shown the answer, so it is stored alone;
                # the question only shapes the embedding that finds it.
                conn.execute(
                    insert(KbChunk).values(
                        text=reply,
                        title="Answered by a health worker",
                        source_type="health_worker_answer",
                        embedding=self.embed([f"{case.details}\n{reply}"])[0],
                    )
                )
            # A prescription is explained to the patient by the document agent
            # once confirmed; the ASHA's raw reply is not sent on.
            if case.cause != "prescription_confirmation":
                self.relay(conn, case, reply)
            conn.execute(
                text(
                    "UPDATE cases SET status = 'answered', asha_reply = :reply, "
                    "answered_at = now() WHERE number = :n"
                ),
                {"reply": reply, "n": number},
            )

    def relay(self, conn: Connection, case, reply: str) -> None:
        if in_window(conn, case.patient_id):
            prefix = in_language(REPLY_PREFIX, case.language, case.script)
            send_reply(
                self.whatsapp,
                self.sarvam,
                case.wa_id,
                f"{prefix} {reply}",
                case.language,
            )
        else:
            self.whatsapp.send_template(
                case.wa_id,
                "case_update",
                TEMPLATE_LANGUAGES.get(case.language, "hi"),
                [one_line(reply)],
            )
