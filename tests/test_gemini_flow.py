"""A signed webhook through real Postgres, Gemini responses, and WhatsApp output."""

import pytest
from test_qa import PATIENT, add_chunk, setup
from test_qa import world as world
from test_webhook import client as client
from test_webhook import delivery, post


@pytest.mark.parametrize("kind", ["text", "audio"])
@pytest.mark.parametrize(
    "cited", [True, False], ids=["grounded-reply", "uncited-answer-escalates"]
)
def test_a_webhook_question_uses_gemini_and_keeps_the_citation_boundary(
    client, world, cited, kind
):
    setup(world)
    chunk = add_chunk(world, "ORS replaces water and salts lost during diarrhoea")
    answer = "ORS helps replace lost water and salts"
    world.language = ("en-IN", "Latn")
    world.chat_replies = ["ORS diarrhoea", answer + (f" [{chunk}]." if cited else ".")]
    world.transcript = "What is ORS used for?"
    world.transcript_language = "en-IN"
    payload = delivery(
        {
            "from": PATIENT,
            "id": "wamid.gemini-e2e",
            "type": kind,
            kind: {"body": "What is ORS used for?"}
            if kind == "text"
            else {"id": "media-synthetic", "mime_type": "audio/ogg"},
        }
    )
    assert post(client, payload).status_code == 200
    assert world.worker.work_one()
    [reply] = world.texts_to(PATIENT)
    if cited:
        assert "replaces" in reply or "replace" in reply
        assert "MoHFW" in reply
        assert world.voice_notes == [PATIENT]
    else:
        assert answer not in reply
        assert [case.cause for case in world.cases()] == ["citation_failed"]
    assert len(world.chat_requests) == 2
    assert post(client, payload).status_code == 200
    assert not world.worker.work_one()  # Meta's retry cannot send another reply.


def test_a_prescription_webhook_waits_for_asha_confirmation_with_gemini(client, engine):
    from sqlalchemy import text
    from test_document import ASHA, LEXICON, RX, RX_MEDICINES, extracted
    from test_document import setup as setup_document
    from test_worker import World

    from chhaaya.document import DocumentAgent

    world = World(engine)
    setup_document(world)
    world.agents["document"] = DocumentAgent(
        world.worker.gemini, world.worker.whatsapp, world.cases_, lambda: LEXICON
    ).read
    world.agents["escalation"] = world.cases_.answer
    world.ocr = RX
    world.chat_replies = [extracted("prescription", medicines=RX_MEDICINES)]
    assert (
        post(
            client,
            delivery(
                {
                    "from": PATIENT,
                    "id": "wamid.gemini-prescription",
                    "type": "image",
                    "image": {"id": "synthetic-photo", "mime_type": "image/jpeg"},
                }
            ),
        ).status_code
        == 200
    )
    assert world.worker.work_one()
    assert all("Dolo" not in reply for reply in world.texts_to(PATIENT))
    assert [case.cause for case in world.cases()] == ["prescription_confirmation"]
    assert world.images[0][0] == ASHA
    with engine.connect() as conn:
        case_id = conn.execute(text("SELECT number FROM cases")).scalar_one()
    assert (
        post(
            client,
            delivery(
                {
                    "from": ASHA,
                    "id": "wamid.gemini-confirmation",
                    "type": "text",
                    "text": {"body": f"{case_id} ok"},
                }
            ),
        ).status_code
        == 200
    )
    assert world.worker.work_one()
    assert any("Dolo" in reply for reply in world.texts_to(PATIENT))
    assert world.cases()[0].status == "answered"
