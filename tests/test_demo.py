"""The browser demo over HTTP: real Postgres and worker, mocked provider boundary."""

import time
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import Engine
from test_document import LEXICON, RX, RX_MEDICINES, extracted
from test_qa import add_chunk
from test_retrieval import fake_embed
from test_worker import World

from chhaaya.cases import NOTICE
from chhaaya.danger_signs import URGENT_REPLY
from chhaaya.demo import DemoSettings, create_app

KEY = "synthetic-demo-key"
RX_PHOTO = (Path(__file__).parents[1] / "eval/sets/documents/rx-01.jpg").read_bytes()


@pytest.fixture
def world(engine: Engine) -> World:
    world = World(engine, speech_enabled=False)
    world.media = RX_PHOTO
    world.language = ("hi-IN", "Latn")
    return world


def demo(world: World) -> TestClient:
    app = create_app(world.engine, world.worker.ai, fake_embed, lambda: LEXICON, KEY)
    return TestClient(app, base_url="http://127.0.0.1")


def sign_in(client: TestClient) -> None:
    response = client.get(f"/?key={KEY}", follow_redirects=False)
    assert response.status_code == 303


def state(client: TestClient) -> dict:
    response = client.get("/api/state")
    assert response.status_code == 200
    return response.json()


def until(client: TestClient, done: Callable[[dict], bool]) -> dict:
    """Poll the state the page polls until `done`, as the browser would."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        current = state(client)
        if done(current):
            return current
        time.sleep(0.05)
    raise AssertionError(f"timed out; last state: {current}")


def settled(current: dict) -> bool:
    inbound = [m for m in current["patient"] + current["asha"] if m["status"]]
    return all(m["status"] in ("done", "failed") for m in inbound)


def send(client: TestClient, role: str, text: str, client_id: str) -> dict:
    response = client.post(
        "/api/messages", json={"role": role, "text": text, "client_id": client_id}
    )
    assert response.status_code == 200, response.text
    return response.json()


def outbound(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m["direction"] == "outbound"]


def test_the_browser_demo_runs_the_pipeline_and_keeps_every_reply(world: World):
    guideline = add_chunk(world, "Give ORS after every loose stool")
    with demo(world) as client:
        # Only the link printed at startup opens the demo.
        assert client.get("/api/state").status_code == 401
        assert client.get("/?key=wrong", follow_redirects=False).status_code == 401
        sign_in(client)
        assert state(client)["assistant_running"]
        foreign = client.get("/api/state", headers={"host": "chhaaya.example"})
        assert foreign.status_code == 400

        # Empty input is refused; a double submit is stored once.
        for blank in ("", "   \n"):
            response = client.post(
                "/api/messages",
                json={"role": "patient", "text": blank, "client_id": "blank-0001"},
            )
            assert response.status_code == 422
        world.chat_replies = [
            "ORS for diarrhoea",
            f"Har dast ke baad ORS dein [{guideline}].",
        ]
        first = send(client, "patient", "bachche ko loose motion hai", "question-1")
        again = send(client, "patient", "bachche ko loose motion hai", "question-1")
        assert again == {**first, "duplicate": True}
        current = until(client, settled)
        [question] = [m for m in current["patient"] if m["direction"] == "inbound"]
        assert question["status"] == "done"
        [answer] = outbound(current["patient"])
        assert answer["body"].startswith("Har dast ke baad ORS dein.")
        assert "Source: Intensified Diarrhoea Control Fortnight" in answer["body"]
        assert not answer["urgent"]

        # The provider fails: the message fails visibly, and a retry is new.
        world.language = None
        send(client, "patient", "zinc kab tak dena hai", "question-2")
        current = until(client, settled)
        failed = current["patient"][-1]
        assert failed["status"] == "failed" and "500" in failed["error"]
        assert len(outbound(current["patient"])) == 1

        # A danger sign is answered from the original text with the provider down.
        send(client, "patient", "bachcha doodh nahi pee raha", "emergency-1")
        current = until(client, settled)
        urgent = outbound(current["patient"])[-1]
        assert urgent["urgent"] and URGENT_REPLY["hi-Latn"] in urgent["body"]
        assert [(c["cause"], c["status"]) for c in current["cases"]] == [
            ("danger sign", "open")
        ]

        # A prescription photo reaches the ASHA; the patient hears nothing of it.
        world.language = ("hi-IN", "Latn")
        send(client, "asha", "namaste", "asha-hello")  # she wrote today
        until(client, settled)
        assert (
            client.post(
                "/api/photos?client_id=gif-photo-1", content=b"GIF89a"
            ).status_code
            == 415
        )
        world.ocr = RX
        world.chat_replies = [extracted("prescription", medicines=RX_MEDICINES)]
        for _ in range(2):
            photo = client.post("/api/photos?client_id=photo-rx-1", content=RX_PHOTO)
            assert photo.status_code == 200
        current = until(client, settled)
        assert [m["kind"] for m in current["patient"]].count("image") == 1
        [notice] = [m["body"] for m in outbound(current["patient"])][-1:]
        assert notice in NOTICE.values()
        assert all("Dolo" not in m["body"] for m in outbound(current["patient"]))
        [case] = [
            c for c in current["cases"] if c["cause"] == "prescription to confirm"
        ]
        [shown] = [m for m in current["asha"] if m["kind"] == "image"]
        assert client.get(shown["photo"]).content == RX_PHOTO
        assert any("Dolo 650: 1-0-1, 5 days" in m["body"] for m in current["asha"])

        send(client, "asha", f"{case['number']} ok", "asha-confirm")
        current = until(client, settled)
        confirmed = outbound(current["patient"])[-1]["body"]
        assert "Dolo 650: 1-0-1, 5 days" in confirmed
        assert "Metfornin 500: 0-0-1, 1 month" in confirmed
        before = state(client)

    # A refresh, even after a restart, shows the same conversation from Postgres.
    with demo(world) as client:
        sign_in(client)
        after = state(client)
        assert after["patient"] == before["patient"]
        assert after["cases"] == before["cases"]


def test_the_demo_refuses_the_whatsapp_database():
    def settings(demo_database_url: str) -> DemoSettings:
        return DemoSettings(
            _env_file=None,
            demo_database_url=demo_database_url,
            database_url="postgresql://chhaaya:pw@localhost:5432/chhaaya",
            anthropic_api_key="synthetic",
        )

    with pytest.raises(ValidationError, match="DEMO_DATABASE_URL"):
        settings("postgresql://chhaaya:other@127.0.0.1/chhaaya")
    # Requiring a distinct database name also covers host aliases and port forwards.
    for url in (
        "postgresql://chhaaya:pw@LOCALHOST:5432/chhaaya",
        "postgresql://chhaaya:pw@localhost:5433/chhaaya",
    ):
        with pytest.raises(ValidationError, match="DEMO_DATABASE_URL"):
            settings(url)
    assert not settings(
        "postgresql://chhaaya:pw@localhost:5433/chhaaya_demo"
    ).speech_enabled


def test_shared_service_settings_are_known_but_misspelled_demo_settings_fail(
    tmp_path: Path, monkeypatch
):
    # Shared service keys must load, while typos must not silently select Claude.
    monkeypatch.chdir(tmp_path)
    values = (
        "DEMO_DATABASE_URL=postgresql://chhaaya:pw@localhost/chhaaya_demo\n"
        "ANTHROPIC_API_KEY=synthetic\nWHATSAPP_ACCESS_TOKEN=synthetic-unused\n"
    )
    env_file = tmp_path / ".env"
    env_file.write_text(values)
    assert DemoSettings().ai_provider == "claude"
    env_file.write_text(values + "AI_PROVDER=sarvam\n")
    with pytest.raises(ValidationError, match="ai_provder"):
        DemoSettings()
