"""The browser demo: the worker's pipeline on a dedicated database, replies kept local.

`uv run python -m chhaaya.demo` serves one page on 127.0.0.1 for a synthetic
patient and ASHA. Typed messages and photos are queued as the webhook queues
them and handled by the same worker and agents. BrowserWhatsApp stores each
reply in the demo database instead of sending it, so nothing reaches Meta or a
phone.
"""

import argparse
import functools
import hmac
import logging
import secrets
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Annotated, Literal, Self

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine, make_url, select, text
from sqlalchemy.dialects.postgresql import insert
from starlette.middleware.trustedhost import TrustedHostMiddleware

from chhaaya import worker
from chhaaya.ai import AIClient
from chhaaya.cases import LABELS
from chhaaya.danger_signs import URGENT_REPLY
from chhaaya.db import Message, User, sqlalchemy_url
from chhaaya.images import image_media_type
from chhaaya.lexicon import Lexicon
from chhaaya.retrieval import Embed
from chhaaya.settings import Settings
from chhaaya.whatsapp import WhatsAppClient

log = logging.getLogger("chhaaya.demo")
PAGE = Path(__file__).with_name("demo.html")
COOKIE = "chhaaya_demo"
# Synthetic ids that are not phone numbers, so no reply could ever reach a phone.
WA_IDS = {"patient": "demo-patient", "asha": "demo-asha"}
TEXT_LIMIT = 4096  # Meta's limit for a text body
PHOTO_LIMIT = 5 * 1024 * 1024  # Meta's limit for an image
# How long a photo remains accessible to the ASHA, in seconds.
PHOTO_LIFETIME = 30 * 60
CLIENT_ID = r"^[A-Za-z0-9-]{8,64}$"
HISTORY = 200
CSP = (
    "default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
    "img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
SIGN_IN = (
    "<!doctype html><meta charset=utf-8><title>Chhaaya demo</title>"
    "<p>Open the link printed in the terminal where the demo started.</p>"
)


class DemoSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="forbid", hide_input_in_errors=True
    )

    demo_database_url: str
    # The WhatsApp service's database, which the demo refuses to share.
    database_url: str | None = None
    ai_provider: Literal["claude", "sarvam"] = "claude"
    anthropic_api_key: str = ""
    sarvam_api_key: str = ""

    @model_validator(mode="before")
    @classmethod
    def shared_service_fields(cls, values: dict) -> dict:
        # The shared .env contains known service fields the browser never uses.
        unused = Settings.model_fields.keys() - cls.model_fields.keys()
        return {name: value for name, value in values.items() if name not in unused}

    @property
    def speech_enabled(self) -> bool:
        """Speech stays off in the demo, whatever SPEECH_ENABLED says."""
        return False

    @model_validator(mode="after")
    def checked(self) -> Self:
        claude = self.ai_provider == "claude"
        key = self.anthropic_api_key if claude else self.sarvam_api_key
        if not key.strip():
            name = "ANTHROPIC_API_KEY" if claude else "SARVAM_API_KEY"
            raise ValueError(f"{name} is required for AI_PROVIDER={self.ai_provider}")
        if (
            self.database_url
            and make_url(self.demo_database_url).database
            == make_url(self.database_url).database
        ):
            # A distinct name also prevents aliases or port forwards sharing the queue.
            raise ValueError("DEMO_DATABASE_URL needs a distinct database name")
        return self


@dataclass
class Photos:
    """Photos in this process's memory only, never on disk or in the database."""

    lifetime: float = PHOTO_LIFETIME
    held: dict[str, tuple[float, bytes]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def put(self, photo: bytes) -> str:
        media_id = f"browser-photo-{secrets.token_urlsafe(16)}"
        with self.lock:
            self.expire()
            self.held[media_id] = (monotonic() + self.lifetime, photo)
        return media_id

    def get(self, media_id: str) -> bytes | None:
        with self.lock:
            self.expire()
            found = self.held.get(media_id)
        return found[1] if found else None

    def take(self, media_id: str) -> bytes:
        """The photo, which is then dropped: the worker reads an inbound one once."""
        with self.lock:
            self.expire()
            found = self.held.pop(media_id, None)
        if found is None:
            raise LookupError("the photo is no longer in memory; send it again")
        return found[1]

    def drop(self, media_id: str) -> None:
        with self.lock:
            self.held.pop(media_id, None)

    def expire(self) -> None:
        now = monotonic()
        for media_id in [m for m, (until, _) in self.held.items() if until < now]:
            del self.held[media_id]


class BrowserWhatsApp(WhatsAppClient):
    """Stores each outbound message in the demo database instead of sending it.

    It overrides every WhatsAppClient method and never creates the HTTP client,
    so no call can reach graph.facebook.com.
    """

    def __init__(self, engine: Engine, photos: Photos) -> None:
        self.engine = engine
        self.photos = photos

    def send_text(self, to: str, body: str) -> str:
        return self.store(to, "text", body)

    def send_audio(self, to: str, media_id: str) -> str:
        raise RuntimeError("speech is off in the browser demo")

    def send_image(self, to: str, media_id: str, caption: str) -> str:
        return self.store(to, "image", caption, media_id)

    def send_template(
        self, to: str, name: str, language: str, params: list[str]
    ) -> str:
        return self.store(to, "template", f"[{name}] " + " | ".join(params))

    def upload_media(self, data: bytes, mime_type: str) -> str:
        image_media_type(data)  # only prescription photos are ever uploaded
        return self.photos.put(data)

    def download_media(self, media_id: str) -> bytes:
        return self.photos.take(media_id)

    def _send(self, to: str, message: dict) -> str:
        raise RuntimeError("the browser demo never sends to WhatsApp")

    def store(self, to: str, kind: str, body: str, media_id: str | None = None) -> str:
        wa_message_id = f"browser-out-{uuid.uuid4().hex}"
        with self.engine.begin() as conn:
            user_id = conn.execute(select(User.id).where(User.wa_id == to)).scalar_one()
            conn.execute(
                insert(Message).values(
                    wa_message_id=wa_message_id,
                    user_id=user_id,
                    direction="outbound",
                    kind=kind,
                    body=body,
                    media_id=media_id,
                )
            )
        return wa_message_id


class Typed(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["patient", "asha"]
    text: str = Field(min_length=1, max_length=TEXT_LIMIT)
    # Made by the page for each message, so a repeated submit is stored once.
    client_id: str = Field(pattern=CLIENT_ID)


def demo_users(engine: Engine) -> dict[str, int]:
    """The synthetic ASHA and her patient, created on first start."""
    users = {}
    with engine.begin() as conn:
        for role in ("asha", "patient"):
            wa_id = WA_IDS[role]
            # The no-op update makes RETURNING give back the id of an existing user.
            users[role] = conn.execute(
                insert(User)
                .values(wa_id=wa_id, role=role, asha_id=users.get("asha"))
                .on_conflict_do_update(index_elements=["wa_id"], set_={"wa_id": wa_id})
                .returning(User.id)
            ).scalar_one()
    return users


def queue(
    engine: Engine,
    user_id: int,
    client_id: str,
    kind: str,
    body: str | None = None,
    media_id: str | None = None,
) -> dict:
    """Store an inbound message for the worker; a repeated client_id is a no-op."""
    wa_message_id = f"browser-{client_id}"
    with engine.begin() as conn:
        stored = conn.execute(
            insert(Message)
            .values(
                wa_message_id=wa_message_id,
                user_id=user_id,
                direction="inbound",
                kind=kind,
                body=body,
                media_id=media_id,
                status="pending",
            )
            .on_conflict_do_nothing(index_elements=["wa_message_id"])
            .returning(Message.id)
        ).scalar_one_or_none()
        if stored is not None:
            return {"id": stored, "duplicate": False}
        existing = conn.execute(
            select(Message.id).where(Message.wa_message_id == wa_message_id)
        ).scalar_one()
    return {"id": existing, "duplicate": True}


def create_app(
    engine: Engine,
    ai: AIClient,
    embed: Embed,
    lexicon: Callable[[], Lexicon],
    key: str,
) -> FastAPI:
    """The demo page and its API, with the worker running in a thread."""
    photos = Photos()
    work, due = worker.build(
        engine, BrowserWhatsApp(engine, photos), ai, embed, lexicon
    )
    users = demo_users(engine)
    stop = threading.Event()
    loop = threading.Thread(
        target=worker.run, args=(work, due, stop), name="demo-worker", daemon=True
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loop.start()
        yield
        stop.set()
        await run_in_threadpool(loop.join)

    app = FastAPI(title="Chhaaya browser demo", lifespan=lifespan, openapi_url=None)
    # A page on another host name cannot reach the demo through DNS rebinding.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    def has_key(request: Request) -> bool:
        cookie = request.cookies.get(COOKIE, "")
        return hmac.compare_digest(cookie.encode(), key.encode())

    def signed_in(request: Request) -> None:
        if not has_key(request):
            raise HTTPException(401, "Open the link printed when the demo started.")

    api = [Depends(signed_in)]

    @app.get("/", response_class=HTMLResponse)
    def page(
        request: Request, key_: Annotated[str, Query(alias="key")] = ""
    ) -> Response:
        if key_:
            if not hmac.compare_digest(key_.encode(), key.encode()):
                return HTMLResponse(SIGN_IN, 401)
            response = RedirectResponse("/", 303)
            response.set_cookie(COOKIE, key, httponly=True, samesite="strict")
            return response
        if not has_key(request):
            return HTMLResponse(SIGN_IN, 401)
        nonce = secrets.token_urlsafe(16)
        return HTMLResponse(
            PAGE.read_text(encoding="utf-8").replace("{{nonce}}", nonce),
            headers={
                "Content-Security-Policy": CSP.format(nonce=nonce),
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
            },
        )

    @app.get("/api/state", dependencies=api)
    def state() -> dict:
        with engine.connect() as conn:
            return {
                "assistant_running": loop.is_alive(),
                "patient": history(conn, users["patient"], photos),
                "asha": history(conn, users["asha"], photos),
                "cases": cases(conn, users["asha"]),
            }

    @app.post("/api/messages", dependencies=api)
    def typed(message: Typed) -> dict:
        user_id = users[message.role]
        return queue(engine, user_id, message.client_id, "text", body=message.text)

    @app.post("/api/photos", dependencies=api)
    async def photo(
        request: Request,
        client_id: Annotated[str, Query(pattern=CLIENT_ID)],
    ) -> dict:
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > PHOTO_LIMIT:
                raise HTTPException(413, "The photo is larger than 5 MB.")
        try:
            image_media_type(bytes(data))
        except ValueError:
            raise HTTPException(415, "Send a JPEG or PNG photo.") from None
        media_id = photos.put(bytes(data))
        queued = await run_in_threadpool(
            queue, engine, users["patient"], client_id, "image", media_id=media_id
        )
        if queued["duplicate"]:
            photos.drop(media_id)
        return queued

    @app.get("/api/photos/{media_id}", dependencies=api)
    def shown(media_id: str) -> Response:
        found = photos.get(media_id)
        if found is None:
            raise HTTPException(404, "This photo is no longer in memory.")
        return Response(
            found,
            media_type=image_media_type(found),
            headers={"Cache-Control": "no-store"},
        )

    return app


def history(conn, user_id: int, photos: Photos) -> list[dict]:
    """The user's latest messages, oldest first."""
    rows = conn.execute(
        text(
            "SELECT id, direction, kind, body, media_id, status, error, created_at "
            "FROM messages WHERE user_id = :u ORDER BY id DESC LIMIT :n"
        ),
        {"u": user_id, "n": HISTORY},
    ).all()
    return [
        {
            "id": row.id,
            "direction": row.direction,
            "kind": row.kind,
            "body": row.body or "",
            "status": row.status,
            "error": row.error,
            "photo": f"/api/photos/{row.media_id}"
            if row.media_id and photos.get(row.media_id) is not None
            else None,
            "urgent": row.direction == "outbound"
            and any(reply in (row.body or "") for reply in URGENT_REPLY.values()),
            "sent_at": row.created_at.isoformat(),
        }
        for row in reversed(rows)
    ]


def cases(conn, asha_id: int) -> list[dict]:
    rows = conn.execute(
        text(
            "SELECT number, cause, status, details, asha_reply FROM cases "
            "WHERE asha_id = :a ORDER BY number"
        ),
        {"a": asha_id},
    ).all()
    return [
        {
            "number": row.number,
            "cause": LABELS[row.cause],
            "status": row.status,
            "details": row.details or "",
            "asha_reply": row.asha_reply,
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the browser demo locally.")
    parser.add_argument("--port", type=int, default=8100)
    port = parser.parse_args().port
    logging.basicConfig(level=logging.INFO)
    settings = DemoSettings()
    engine = create_engine(sqlalchemy_url(settings.demo_database_url))
    key = secrets.token_urlsafe(24)
    app = create_app(
        engine,
        AIClient(settings),
        worker.lazy_bge_m3(),
        functools.cache(Lexicon.load),
        key,
    )
    print(f"Chhaaya browser demo: http://127.0.0.1:{port}/?key={key}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
