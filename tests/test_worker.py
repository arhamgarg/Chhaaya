import base64
import io
import json
import threading
import time
import zipfile
from collections.abc import Callable
from email.parser import BytesParser
from email.policy import default

import httpx2
import pytest
from sqlalchemy import Engine, text
from test_audio import tone

from chhaaya import reminders
from chhaaya.ai import AIClient
from chhaaya.cases import Cases
from chhaaya.claude import LANGUAGE_PROMPT
from chhaaya.qa import REWRITE
from chhaaya.settings import Settings
from chhaaya.whatsapp import WhatsAppClient
from chhaaya.worker import Inbound, Worker


class World:
    """Meta and the AI providers answer from here, and agents record calls."""

    def __init__(self, engine: Engine, provider="claude", speech_enabled=False):
        self.engine = engine
        self.sent: list[tuple[str, str]] = []
        self.templates: list[tuple[str, str, list[str]]] = []
        self.voice_notes: list[str] = []
        self.language: tuple[str, str] | None = ("hi-IN", "Latn")
        self.transcript: str | None = None  # None means speech-to-text is down
        self.transcript_language: str | None = "hi-IN"
        self.ai_calls: list[str] = []
        # The provider answers each chat request with the next of these, in order.
        self.chat_replies: list[str] = []
        self.rewrite_replies: list[str] = []
        self.chat_reasoning_tokens: list[int] = []
        self.chat_requests: list[dict] = []
        # The Markdown read off a photo; None means OCR fails.
        self.ocr: str | None = None
        self.media = b"OggS-inbound"
        self.media_type = "image/jpeg"
        self.images: list[tuple[str, str]] = []  # (to, caption)
        self.whatsapp_up = True
        self.agent_calls: list[tuple[str, int]] = []
        http = httpx2.Client(transport=httpx2.MockTransport(self.handle))
        self.agents: dict[str, Callable[[Inbound], None]] = {
            name: self.agent(name) for name in ("qa", "document", "escalation")
        }
        whatsapp = WhatsAppClient("token", "123456", http)
        ai = AIClient(
            Settings(
                _env_file=None,
                postgres_password="unused",
                database_url="postgresql://unused",
                whatsapp_verify_token="unused",
                whatsapp_app_secret="unused",
                whatsapp_access_token="unused",
                whatsapp_phone_number_id="unused",
                anthropic_api_key="claude-key",
                sarvam_api_key="sarvam-key"
                if provider == "sarvam" or speech_enabled
                else "",
                ai_provider=provider,
                speech_enabled=speech_enabled,
            ),
            http,
        )
        self.embedded: list[str] = []
        self.cases_ = Cases(engine, whatsapp, ai, self.embed)
        # Most tests mock the classifier boundary; test_intent_flow loads a real one.
        self.worker = Worker(
            engine,
            whatsapp,
            ai,
            self.cases_,
            self.agents,
            lambda body: (
                "reminder_setup" if reminders.ASKS.search(body) else "question"
            ),
        )

    def embed(self, texts):
        """A stand-in for BGE-M3: one fixed 1024-dimensional vector per text."""
        self.embedded.extend(texts)
        return [[1.0] + [0.0] * 1023 for _ in texts]

    def agent(self, name: str) -> Callable[[Inbound], None]:
        return lambda message: self.agent_calls.append((name, message.id))

    def chat_reply(self, system: str, user: str) -> str:
        # Unconfigured rewrites echo the input; generation remains explicit.
        if system == REWRITE:
            return self.rewrite_replies.pop(0) if self.rewrite_replies else user
        return self.chat_replies.pop(0)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if request.url.host == "api.anthropic.com":
            self.ai_calls.append(path.strip("/"))
            return self.claude(request)
        if request.url.host == "api.sarvam.ai":
            self.ai_calls.append(path.strip("/"))
            return self.sarvam(request)
        if request.url.host == "files.sarvam.example":
            return httpx2.Response(200, content=self.vision_output())
        if request.method == "GET":  # an inbound voice note
            if request.url.host == "graph.facebook.com":
                return httpx2.Response(200, json={"url": "https://lookaside.example/m"})
            return httpx2.Response(200, content=self.media)
        if not self.whatsapp_up:
            return httpx2.Response(401, json={"error": {"message": "token expired"}})
        if path.endswith("/media"):
            multipart = BytesParser(policy=default).parsebytes(
                f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
                + request.read()
            )
            parts = {
                part.get_param("name", header="content-disposition"): part
                for part in multipart.iter_parts()
            }
            mime_type = parts["type"].get_content().strip()
            if mime_type.startswith("image/") and (
                mime_type != self.media_type
                or parts["file"].get_content_type() != self.media_type
                or parts["file"].get_payload(decode=True) != self.media
            ):
                return httpx2.Response(400, json={"error": "image type mismatch"})
            return httpx2.Response(200, json={"id": "media-up"})
        body = json.loads(request.read())
        if body["type"] != "template" and not self.in_window(body["to"]):
            # Meta refuses free-form messages outside the 24-hour window.
            error = {"code": 131047, "message": "Re-engagement message"}
            return httpx2.Response(400, json={"error": error})
        if body["type"] == "audio":
            self.voice_notes.append(body["to"])
        elif body["type"] == "image":
            self.images.append((body["to"], body["image"]["caption"]))
        elif body["type"] == "template":
            template = body["template"]
            params = [p["text"] for p in template["components"][0]["parameters"]]
            self.templates.append((body["to"], template["name"], params))
        elif len(body["text"]["body"]) > 4096:  # Meta's limit for a text body
            return httpx2.Response(400, json={"error": {"code": 100}})
        else:
            self.sent.append((body["to"], body["text"]["body"]))
        return httpx2.Response(200, json={"messages": [{"id": "wamid.out"}]})

    def in_window(self, wa_id: str) -> bool:
        with self.engine.connect() as conn:
            return conn.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM messages m JOIN users u "
                    "ON u.id = m.user_id WHERE u.wa_id = :wa_id "
                    "AND m.direction = 'inbound' "
                    "AND m.created_at > now() - interval '24 hours')"
                ),
                {"wa_id": wa_id},
            ).scalar_one()

    def sarvam(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path.endswith("/digitise"):
            body = request.read()
            extension = "jpg" if self.media_type == "image/jpeg" else "png"
            if (
                self.media not in body
                or f"Content-Type: {self.media_type}\r\n".encode() not in body
                or f'filename="photo.{extension}"'.encode() not in body
            ):
                return httpx2.Response(400, json={"error": "image type mismatch"})
            return httpx2.Response(202, json={"job_id": "job-1"})
        if path.endswith("/status"):
            return httpx2.Response(
                200, json={"status": "completed" if self.ocr is not None else "failed"}
            )
        if path.endswith("/download-url"):
            return httpx2.Response(
                200, json={"url": "https://files.sarvam.example/job-1.zip"}
            )
        if path.endswith("text-to-speech"):
            return httpx2.Response(
                200, json={"audios": [base64.b64encode(tone(0.3)).decode()]}
            )
        if path.endswith("speech-to-text"):
            if self.transcript is None:
                return httpx2.Response(500, json={"error": {"message": "down"}})
            return httpx2.Response(
                200,
                json={
                    "transcript": self.transcript,
                    "language_code": self.transcript_language,
                },
            )
        if path.endswith("chat/completions"):
            body = json.loads(request.read())
            self.chat_requests.append(
                {
                    "system": body["messages"][0]["content"],
                    "user": body["messages"][1]["content"],
                }
            )
            reply = self.chat_reply(
                body["messages"][0]["content"], body["messages"][1]["content"]
            )
            reasoning = (
                self.chat_reasoning_tokens.pop(0) if self.chat_reasoning_tokens else 0
            )
            if body.get("reasoning_effort", "medium") is None:
                reasoning = 0
            # Sarvam's default completion budget includes its reasoning tokens.
            complete = body.get("max_tokens", 2048) - reasoning >= len(reply.split())
            return httpx2.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop" if complete else "length",
                            "message": {"content": reply if complete else None},
                        }
                    ]
                },
            )
        if self.language is None:
            return httpx2.Response(500, json={"error": {"message": "down"}})
        language, script = self.language
        return httpx2.Response(
            200, json={"language_code": language, "script_code": script}
        )

    def vision_output(self) -> bytes:
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("photo.md", self.ocr or "")
        return archive.getvalue()

    def claude(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.read())
        if body["system"] == LANGUAGE_PROMPT:
            if self.language is None:
                return httpx2.Response(500, json={"error": {"message": "down"}})
            language, script = self.language
            answer = json.dumps({"language_code": language, "script_code": script})
        elif body["messages"][0]["content"][0]["type"] == "image":
            source = body["messages"][0]["content"][0]["source"]
            if (
                base64.b64decode(source["data"]) != self.media
                or source["media_type"] != self.media_type
            ):
                return httpx2.Response(400, json={"error": "image type mismatch"})
            if self.ocr is None:
                return httpx2.Response(500, json={"error": {"message": "down"}})
            answer = self.ocr
        else:
            self.chat_requests.append(
                {
                    "system": body["system"],
                    "user": body["messages"][0]["content"][0]["text"],
                }
            )
            answer = self.chat_reply(
                body["system"], body["messages"][0]["content"][0]["text"]
            )
        return httpx2.Response(
            200,
            json={
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": answer}],
            },
        )

    def add_user(self, wa_id: str, role: str = "patient", asha: int | None = None):
        with self.engine.begin() as conn:
            return conn.execute(
                text(
                    "INSERT INTO users (wa_id, role, asha_id) "
                    "VALUES (:wa_id, :role, :asha) RETURNING id"
                ),
                {"wa_id": wa_id, "role": role, "asha": asha},
            ).scalar_one()

    def receive(
        self, user: int, kind: str = "text", body: str | None = None, wamid: str = ""
    ) -> int:
        with self.engine.begin() as conn:
            return conn.execute(
                text(
                    "INSERT INTO messages (wa_message_id, user_id, direction, kind, "
                    "body, status) VALUES (:wamid, :user, 'inbound', :kind, :body, "
                    "'pending') RETURNING id"
                ),
                {
                    "wamid": wamid or f"wamid.{time.monotonic_ns()}",
                    "user": user,
                    "kind": kind,
                    "body": body,
                },
            ).scalar_one()

    def message(self, message_id: int):
        with self.engine.connect() as conn:
            return conn.execute(
                text(
                    "SELECT status, error, body, language FROM messages WHERE id = :id"
                ),
                {"id": message_id},
            ).one()

    def user(self, user_id: int):
        with self.engine.connect() as conn:
            return conn.execute(
                text("SELECT language, script FROM users WHERE id = :id"),
                {"id": user_id},
            ).one()

    def texts_to(self, wa_id: str) -> list[str]:
        return [body for to, body in self.sent if to == wa_id]

    def cases(self):
        with self.engine.connect() as conn:
            return conn.execute(
                text("SELECT cause, patient_id, asha_id, status FROM cases")
            ).all()


@pytest.fixture
def world(engine: Engine, request: pytest.FixtureRequest) -> World:
    return World(engine, speech_enabled=bool(request.node.get_closest_marker("speech")))


def test_two_workers_never_take_the_same_message(world: World):
    patient = world.add_user("911111111111")
    ids = [world.receive(patient, body=f"sawal number {n}") for n in range(40)]
    seen: list[int] = []

    def slow_qa(message: Inbound) -> None:
        time.sleep(0.005)
        seen.append(message.id)

    world.agents["qa"] = slow_qa

    def drain() -> None:
        while world.worker.work_one():
            pass

    threads = [threading.Thread(target=drain) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(seen) == ids
    assert all(world.message(i).status == "done" for i in ids)


def test_a_danger_sign_gets_an_urgent_reply_and_a_case_even_if_the_provider_is_down(
    world: World,
):
    world.language = None  # AI provider is down
    asha = world.add_user("919999999999", "asha")
    patient = world.add_user("911111111111", asha=asha)
    message = world.receive(patient, body="pregnancy mein khoon aa raha hai")

    assert world.worker.work_one()

    [(to, reply)] = world.sent
    assert to == "911111111111"
    assert "108" in reply
    assert world.cases() == [("danger_sign", patient, asha, "open")]
    assert world.message(message).status == "done"
    assert world.ai_calls == []
    assert world.agent_calls == []


def test_the_asha_case_opens_even_when_the_urgent_reply_cannot_be_sent(
    world: World,
):
    world.whatsapp_up = False
    asha = world.add_user("919999999999", "asha")
    patient = world.add_user("911111111111", asha=asha)
    message = world.receive(patient, body="usko jhatke aa rahe hain")

    world.worker.work_one()

    assert world.cases() == [("danger_sign", patient, asha, "open")]
    status, error, *_ = world.message(message)
    assert status == "failed"
    assert "401" in error


def test_a_danger_sign_from_a_patient_without_an_asha_is_answered_then_failed(
    world: World,
):
    patient = world.add_user("911111111111")
    message = world.receive(patient, body="My baby is not feeding since morning")

    world.worker.work_one()

    assert len(world.sent) == 1
    assert world.cases() == []
    status, error, *_ = world.message(message)
    assert status == "failed"
    assert "ASHA" in error


def test_typed_text_stores_the_language_and_goes_to_the_qa_agent(world: World):
    patient = world.add_user("911111111111")
    message = world.receive(patient, body="bukhar mein kya khana chahiye")

    world.worker.work_one()

    assert world.user(patient) == ("hi", "Latn")
    assert world.agent_calls == [("qa", message)]
    assert world.message(message).status == "done"
    assert world.sent == []


def test_an_ashas_message_skips_screening_and_a_photo_goes_to_the_document_agent(
    world: World,
):
    asha = world.add_user("919999999999", "asha")
    patient = world.add_user("911111111111", asha=asha)
    reply = world.receive(asha, body="12 usko khoon aa raha tha ab theek hai")
    photo = world.receive(patient, kind="image")

    world.worker.work_one()
    world.worker.work_one()

    assert world.agent_calls == [("escalation", reply), ("document", photo)]
    assert world.sent == []
    assert world.cases() == []
    assert world.ai_calls == []


def test_a_failed_message_keeps_its_error_and_is_not_retried(world: World):
    patient = world.add_user("911111111111")
    message = world.receive(patient, body="sugar kitni honi chahiye")

    def broken(_: Inbound) -> None:
        raise RuntimeError("retrieval is down")

    world.agents["qa"] = broken

    assert world.worker.work_one()
    status, error, *_ = world.message(message)
    assert status == "failed"
    assert "RuntimeError: retrieval is down" in error
    assert not world.worker.work_one()


@pytest.mark.parametrize(
    ("kind", "body", "language", "expected"),
    [
        ("text", "vanakkam", ("ta-IN", "Taml"), "ta-IN"),
        ("text", "hello", None, "api.anthropic.com"),
    ],
)
def test_messages_that_cannot_be_handled_fail_loudly(
    world: World, kind, body, language, expected
):
    world.language = language
    patient = world.add_user("911111111111")
    message = world.receive(patient, kind=kind, body=body)

    world.worker.work_one()

    status, error, *_ = world.message(message)
    assert status == "failed"
    assert expected in error
    assert world.agent_calls == []


def test_a_missing_agent_fails_the_message_instead_of_dropping_it(world: World):
    del world.agents["qa"]
    patient = world.add_user("911111111111")
    message = world.receive(patient, body="bukhar kyun aata hai")

    world.worker.work_one()

    status, error, *_ = world.message(message)
    assert status == "failed"
    assert "qa" in error


def test_nothing_pending_means_nothing_to_do(world: World):
    assert not world.worker.work_one()


def test_a_message_stranded_by_a_dead_worker_is_failed_with_a_reason(world: World):
    patient = world.add_user("911111111111")
    stranded = world.receive(patient, body="pregnancy mein khoon aa raha hai")
    busy = world.receive(patient, body="bukhar kyun aata hai")
    with world.engine.begin() as conn:
        # One worker died 20 minutes into a message; another is working right now.
        conn.execute(
            text(
                "UPDATE messages SET status = 'processing', claimed_at = "
                "CASE WHEN id = :stranded THEN now() - interval '20 minutes' "
                "ELSE now() END"
            ),
            {"stranded": stranded},
        )

    assert not world.worker.work_one()

    status, error, *_ = world.message(stranded)
    assert status == "failed" and "stopped" in error
    assert world.message(busy)[:2] == ("processing", None)
    assert world.sent == []


@pytest.mark.speech
def test_a_voice_note_is_transcribed_and_then_handled_like_typed_text(world: World):
    world.transcript = "bukhar mein kya khana chahiye"
    patient = world.add_user("911111111111")
    message = world.receive(patient, kind="audio")

    world.worker.work_one()

    assert world.message(message) == ("done", None, world.transcript, "hi")
    assert world.user(patient) == ("hi", "Latn")
    assert world.agent_calls == [("qa", message)]


@pytest.mark.speech
def test_a_dangerous_voice_note_gets_the_urgent_reply_as_text_and_a_voice_note(
    world: World,
):
    world.transcript = "pregnancy mein khoon aa raha hai"
    asha = world.add_user("919999999999", "asha")
    patient = world.add_user("911111111111", asha=asha)
    message = world.receive(patient, kind="audio")

    world.worker.work_one()

    [(to, reply)] = world.sent
    assert to == "911111111111" and "108" in reply
    assert world.voice_notes == ["911111111111"]
    assert world.cases() == [("danger_sign", patient, asha, "open")]
    assert world.message(message).status == "done"


@pytest.mark.speech
def test_a_voice_note_that_cannot_be_read_gets_a_text_reply_not_silence(world: World):
    world.transcript = None  # speech-to-text is down
    patient = world.add_user("911111111111")
    message = world.receive(patient, kind="audio")

    world.worker.work_one()

    [(to, reply)] = world.sent
    assert to == "911111111111" and "108" in reply and "type" in reply
    assert world.voice_notes == []
    assert world.message(message).status == "failed"
    assert world.agent_calls == []


@pytest.mark.speech
def test_a_dangerous_voice_note_in_another_language_still_opens_a_case(world: World):
    # A transcript labelled as Marathi must still be screened for danger signs.
    world.transcript = "pregnancy mein khoon aa raha hai"
    world.transcript_language = "mr-IN"
    asha = world.add_user("919999999999", "asha")
    patient = world.add_user("911111111111", asha=asha)
    message = world.receive(patient, kind="audio")

    world.worker.work_one()

    [(to, reply)] = world.sent
    assert to == "911111111111" and "108" in reply
    assert world.cases() == [("danger_sign", patient, asha, "open")]
    assert world.message(message) == ("done", None, world.transcript, None)


@pytest.mark.parametrize(
    ("transcript", "language"), [("vanakkam", "ta-IN"), ("", "hi-IN")]
)
@pytest.mark.speech
def test_a_voice_note_in_another_language_or_silent_asks_the_patient_to_type(
    world: World, transcript: str, language: str
):
    world.transcript = transcript
    world.transcript_language = language
    patient = world.add_user("911111111111")
    message = world.receive(patient, kind="audio")

    world.worker.work_one()

    [(to, reply)] = world.sent
    assert to == "911111111111" and "108" in reply and "type" in reply
    assert world.message(message).status == "failed"
    assert world.user(patient) == (None, None)
    assert world.agent_calls == []
