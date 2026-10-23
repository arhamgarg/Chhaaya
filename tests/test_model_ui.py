"""Model evidence stays private and never blocks the patient conversation."""

import threading
from concurrent.futures import ThreadPoolExecutor

from test_demo import demo, sign_in, state
from test_worker import World


def test_model_evidence_and_slow_explanations_leave_the_conversation_available(
    engine, monkeypatch
):
    world = World(engine, speech_enabled=False)
    with demo(world) as client:
        for path in ("/models", "/api/models"):
            assert client.get(path).status_code == 401
        assert (
            client.post("/api/models/explain", json={"text": "hello"}).status_code
            == 401
        )

        from chhaaya import model_metrics

        saved = {
            "schema_version": 1,
            "test": {"rows": 120},
            "models": [{"id": "muril"}],
        }
        monkeypatch.setattr(model_metrics, "load_report", lambda: saved)
        sign_in(client)
        page = client.get("/models")
        assert page.status_code == 200
        assert "Content-Security-Policy" in page.headers
        assert client.get("/api/models").json() == saved
        for invalid in (
            {"text": " "},
            {"text": "x" * 513},
            {"text": "hi", "train": True},
        ):
            assert client.post("/api/models/explain", json=invalid).status_code == 422

        started, finish = threading.Event(), threading.Event()

        def explain(text):
            started.set()
            assert finish.wait(5)
            return {"text": text, "routed": "smalltalk", "tokens": ["hello"]}

        monkeypatch.setattr(model_metrics, "explain_message", explain)
        before = state(client)
        with ThreadPoolExecutor() as executor:
            pending = executor.submit(
                client.post, "/api/models/explain", json={"text": "hello"}
            )
            assert started.wait(2)
            try:
                current = executor.submit(state, client).result(timeout=2)
                assert current["patient"] == before["patient"]
                assert current["cases"] == before["cases"]
                assert current["assistant_running"]
                assert client.get("/api/models").json() == saved
            finally:
                finish.set()
            assert pending.result().json() == {
                "text": "hello",
                "routed": "smalltalk",
                "tokens": ["hello"],
            }

        def missing():
            raise FileNotFoundError("Run uv run python -m chhaaya.model_metrics first.")

        monkeypatch.setattr(model_metrics, "load_report", missing)
        response = client.get("/api/models")
        assert response.status_code == 503
        assert "model_metrics" in response.json()["detail"]
        assert state(client)["patient"] == before["patient"]
