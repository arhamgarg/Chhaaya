"""Offline browser fixture: `uv run python tests/browser_demo_server.py`.

Uses a disposable Postgres database and HTTP provider mocks from the pipeline
suite. Open http://127.0.0.1:8127/?key=synthetic-demo-key, then run
`node tests/browser_demo_retry.cjs PLAYWRIGHT_MODULE CHROMIUM OUTPUT_JSON`.
The arguments name an existing Playwright module and Chromium executable.
No provider or Meta request leaves the process.
"""

import pytest
import uvicorn
from conftest import migrate
from fastapi.responses import Response
from sqlalchemy import create_engine
from test_demo import KEY, RX_PHOTO
from test_document import LEXICON, RX, RX_MEDICINES, extracted
from test_retrieval import fake_embed
from test_worker import World
from testcontainers.community.postgres import PostgresContainer

from chhaaya.db import sqlalchemy_url
from chhaaya.demo import create_app


def main():
    with PostgresContainer("pgvector/pgvector:pg16", driver=None) as postgres:
        url = postgres.get_connection_url()
        with pytest.MonkeyPatch.context() as monkeypatch:
            migrate(url, monkeypatch, "head")
            engine = create_engine(sqlalchemy_url(url))
            try:
                world = World(engine, speech_enabled=False)
                world.language = ("hi-IN", "Latn")
                world.media = RX_PHOTO
                world.ocr = RX
                world.chat_replies = [
                    extracted("prescription", medicines=RX_MEDICINES)
                ] * 20
                app = create_app(
                    engine,
                    world.worker.ai,
                    fake_embed,
                    lambda: LEXICON,
                    KEY,
                    lambda _: "smalltalk",
                )

                @app.get("/test-photo")
                def photo(different: bool = False):
                    world.media = RX_PHOTO + (b"\x00" if different else b"")
                    return Response(world.media, media_type="image/jpeg")

                uvicorn.run(app, host="127.0.0.1", port=8127)
            finally:
                engine.dispose()


if __name__ == "__main__":
    main()
