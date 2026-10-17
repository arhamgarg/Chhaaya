from datetime import timedelta

import pytest
from sqlalchemy import Engine, text
from test_worker import World

ASHA = "919999999999"
PATIENT = "911111111111"


@pytest.fixture
def world(engine: Engine) -> World:
    world = World(engine)
    world.agents["escalation"] = world.cases_.answer
    return world


def setup(world: World):
    asha = world.add_user(ASHA, "asha")
    patient = world.add_user(PATIENT, asha=asha)
    return asha, patient


def silent_for(world: World, user: int, hours: int) -> None:
    """Pretend the user's last message arrived `hours` ago."""
    with world.engine.begin() as conn:
        conn.execute(
            text("UPDATE messages SET created_at = now() - :age WHERE user_id = :u"),
            {"age": timedelta(hours=hours), "u": user},
        )


def case(world: World, number: int):
    with world.engine.connect() as conn:
        return conn.execute(
            text("SELECT status, asha_reply FROM cases WHERE number = :n"),
            {"n": number},
        ).one()


def chunks(world: World):
    with world.engine.connect() as conn:
        return conn.execute(text("SELECT text, source_type FROM kb_chunks")).all()


def test_a_question_case_alerts_the_asha_tells_the_patient_and_relays_the_answer(
    world: World,
):
    asha, patient = setup(world)
    world.receive(asha, body="namaste")  # the ASHA is inside the 24-hour window
    world.receive(patient, body="kya ORS roz de sakte hain")
    world.worker.work_one()
    world.worker.work_one()  # the ASHA's greeting gets the usage hint
    world.sent.clear()

    number = world.cases_.open("weak_evidence", patient, "kya ORS roz de sakte hain")

    [alert] = world.texts_to(ASHA)
    assert f"{number} " in alert and "kya ORS roz de sakte hain" in alert
    assert "personal details" in alert  # the answer will be shared
    [notice] = world.texts_to(PATIENT)
    assert "ASHA" in notice

    world.sent.clear()
    reply = world.receive(asha, body=f"{number} Haan, dast mein ORS roz dein")
    world.worker.work_one()

    [relayed] = world.texts_to(PATIENT)
    assert "Haan, dast mein ORS roz dein" in relayed
    assert world.voice_notes[-1] == PATIENT
    assert case(world, number) == ("answered", "Haan, dast mein ORS roz dein")
    assert world.message(reply).status == "done"
    # Only the answer is shared with later patients, never this patient's words.
    assert chunks(world) == [("Haan, dast mein ORS roz dein", "health_worker_answer")]
    assert (
        world.embedded[-1] == "kya ORS roz de sakte hain\nHaan, dast mein ORS roz dein"
    )


def test_outside_the_24_hour_window_both_sides_get_templates(world: World):
    asha, patient = setup(world)
    world.receive(patient, body="sawal")
    silent_for(world, patient, hours=30)

    number = world.cases_.open("critical_lab_value", patient, "Hb 5.1 g/dL")

    assert world.texts_to(PATIENT) == []  # no template carries the notice
    [(to, name, params)] = world.templates
    assert (to, name) == (ASHA, "asha_case_alert")
    assert str(number) in params and "Hb 5.1 g/dL" in params

    world.receive(asha, body=f"{number} Kal subah PHC aayiye")
    world.worker.work_one()
    world.worker.work_one()

    assert world.templates[-1] == (PATIENT, "case_update", ["Kal subah PHC aayiye"])
    assert case(world, number)[0] == "answered"
    assert chunks(world) == []  # only answers to questions become knowledge


@pytest.mark.parametrize(
    "body", ["theek hai", "case 12 ho gaya", "12", "   ", "abc 12 answer"]
)
def test_an_unparseable_asha_reply_gets_a_usage_hint(world: World, body: str):
    asha, _ = setup(world)
    message = world.receive(asha, body=body)

    world.worker.work_one()

    [hint] = world.texts_to(ASHA)
    assert "12" in hint
    assert world.message(message).status == "done"


def test_an_asha_cannot_answer_a_case_that_is_not_hers_or_is_closed(world: World):
    asha, patient = setup(world)
    other = world.add_user("918888888888", "asha")
    world.receive(patient, body="sawal")
    number = world.cases_.open("weak_evidence", patient, "sawal")
    world.sent.clear()

    world.receive(other, body=f"{number} galat jawab")
    world.worker.work_one()
    world.worker.work_one()

    assert case(world, number) == ("open", None)
    assert world.texts_to(PATIENT) == []
    [refusal] = world.texts_to("918888888888")
    assert str(number) in refusal

    world.receive(asha, body=f"{number} pehla jawab")
    world.receive(asha, body=f"{number} doosra jawab")
    world.worker.work_one()
    world.worker.work_one()

    assert case(world, number) == ("answered", "pehla jawab")
    assert len(world.texts_to(PATIENT)) == 1
    assert chunks(world) == [("pehla jawab", "health_worker_answer")]
    [refusal] = world.texts_to(ASHA)  # the alert went out as a template
    assert str(number) in refusal


def test_a_prescription_case_accepts_ok_or_corrections_without_telling_the_patient(
    world: World,
):
    asha, patient = setup(world)
    world.receive(patient, body="parcha")
    first = world.cases_.open("prescription_confirmation", patient, "Amlodipine 5mg")
    second = world.cases_.open("prescription_confirmation", patient, "Metformin")
    world.sent.clear()

    world.receive(asha, body=f"{first} ok")
    world.receive(asha, body=f"{second} Metformin 500mg do baar, khane ke baad")
    for _ in range(3):
        world.worker.work_one()

    assert case(world, first) == ("answered", "ok")
    assert case(world, second) == ("answered", "Metformin 500mg do baar, khane ke baad")
    assert world.texts_to(PATIENT) == []


def test_if_the_relay_fails_the_case_stays_open_for_another_try(world: World):
    asha, patient = setup(world)
    world.receive(patient, body="sawal")
    number = world.cases_.open("weak_evidence", patient, "sawal")
    world.whatsapp_up = False

    reply = world.receive(asha, body=f"{number} jawab")
    world.worker.work_one()
    world.worker.work_one()

    assert case(world, number) == ("open", None)
    assert chunks(world) == []
    assert world.message(reply).status == "failed"


def test_a_danger_sign_case_alerts_the_asha_without_a_second_patient_notice(
    world: World,
):
    asha, patient = setup(world)
    world.receive(patient, body="usko jhatke aa rahe hain")

    world.worker.work_one()

    assert len(world.texts_to(PATIENT)) == 1  # only the urgent reply
    [(to, name, params)] = world.templates  # the ASHA has not written today
    assert (to, name) == (ASHA, "asha_case_alert")
    assert any("jhatke" in p for p in params)


def test_a_patient_without_an_asha_cannot_have_a_case(world: World):
    patient = world.add_user(PATIENT)
    with pytest.raises(LookupError, match="ASHA"):
        world.cases_.open("weak_evidence", patient, "sawal")


@pytest.mark.parametrize("asha_wrote_today", [True, False])
def test_a_long_danger_sign_message_still_reaches_the_asha(
    world: World, asha_wrote_today: bool
):
    asha, patient = setup(world)
    if asha_wrote_today:
        world.receive(asha, body="namaste")
        world.worker.work_one()
        world.sent.clear()
    world.receive(
        patient, body="usko jhatke aa rahe hain. " + "bahut dar lag raha hai " * 150
    )

    world.worker.work_one()

    if asha_wrote_today:
        [alert] = world.texts_to(ASHA)
        assert "jhatke" in alert and len(alert) <= 1024
    else:
        [(to, _, params)] = world.templates
        assert to == ASHA and any("jhatke" in p for p in params)
        assert sum(len(p) for p in params) <= 900  # body limit 1024 with its text
    with world.engine.connect() as conn:
        details = conn.execute(text("SELECT details FROM cases")).scalar_one()
    assert len(details) > 3000  # the full message stays on the case
