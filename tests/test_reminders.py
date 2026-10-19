import json
from datetime import UTC, date, datetime, time, timedelta

import pytest
from sqlalchemy import Engine, text
from test_worker import World

from chhaaya.reminders import IST, Reminders, next_due

ASHA = "919999999999"
PATIENT = "911111111111"


class ReminderWorld(World):
    def __init__(self, engine: Engine):
        super().__init__(engine)
        worker = self.worker
        self.reminders = Reminders(engine, worker.whatsapp, worker.sarvam, self.cases_)
        self.agents["reminder"] = self.reminders.handle
        self.agents["escalation"] = self.cases_.answer


@pytest.fixture
def world(engine: Engine) -> ReminderWorld:
    return ReminderWorld(engine)


def setup(world: ReminderWorld) -> tuple[int, int]:
    asha = world.add_user(ASHA, "asha")
    patient = world.add_user(PATIENT, asha=asha)
    return asha, patient


def say(
    world: ReminderWorld, patient: int, body: str, *replies: str, kind="text"
) -> int:
    world.chat_replies = list(replies)
    if kind == "audio":
        world.transcript = body
        message = world.receive(patient, kind="audio")
    else:
        message = world.receive(patient, body=body)
    world.worker.work_one()
    return message


def extracted(*reminders: tuple[str, list[str], int | None]) -> str:
    return json.dumps(
        {
            "reminders": [
                {"medicine": medicine, "times": times, "days": days}
                for medicine, times, days in reminders
            ]
        }
    )


def reminders(world: ReminderWorld):
    with world.engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT id, medicine, times, starts_on, ends_on, next_due_at, "
                "confirmed_at FROM reminders ORDER BY id"
            )
        ).all()


def doses(world: ReminderWorld):
    with world.engine.connect() as conn:
        return conn.execute(
            text("SELECT reminder_id, status FROM doses ORDER BY due_at")
        ).all()


def make_due(
    world: ReminderWorld, reminder_id: int, ago: timedelta = timedelta(minutes=1)
):
    with world.engine.begin() as conn:
        conn.execute(
            text("UPDATE reminders SET next_due_at = now() - :ago WHERE id = :id"),
            {"ago": ago, "id": reminder_id},
        )


def saved_reminder(
    world: ReminderWorld, patient: int, times=("08:00",), days=None
) -> int:
    say(
        world,
        patient,
        "metformin yaad dilana",
        extracted(("Metformin", list(times), days)),
    )
    say(world, patient, "haan")
    world.sent.clear()
    return reminders(world)[-1].id


@pytest.mark.parametrize("kind", ["text", "audio"])
def test_a_reminder_is_read_back_and_saved_only_after_a_yes(
    world: ReminderWorld, kind: str
):
    _, patient = setup(world)
    request = "subah 8 baje aur raat 8 baje metformin yaad dilana, ek mahine tak"

    say(
        world,
        patient,
        request,
        extracted(("Metformin", ["08:00", "20:00"], 30)),
        kind=kind,
    )

    # The LLM read the patient's own words, and nothing is scheduled yet.
    assert world.chat_requests[0]["messages"][1]["content"] == request
    [read_back] = world.texts_to(PATIENT)
    assert (
        "Metformin" in read_back and "8:00 am" in read_back and "8:00 pm" in read_back
    )
    assert "30" in read_back
    [draft] = reminders(world)
    assert draft.next_due_at is None and draft.confirmed_at is None

    say(world, patient, "haan ji")

    [saved] = reminders(world)
    assert saved.times == [time(8), time(20)]  # the patient's times, in IST
    assert saved.ends_on - saved.starts_on == timedelta(days=29)
    # Stored in UTC: 08:00 IST is 02:30 UTC, 20:00 IST is 14:30 UTC.
    due = saved.next_due_at.astimezone(UTC)
    assert due.time() in (time(2, 30), time(14, 30))
    assert timedelta(0) < due - datetime.now(UTC) <= timedelta(hours=12)
    assert len(world.texts_to(PATIENT)) == 2  # the read-back, then "saved"


def test_a_no_discards_the_reminder(world: ReminderWorld):
    _, patient = setup(world)

    say(
        world,
        patient,
        "BP ki goli raat 9 baje yaad dilana",
        extracted(("BP", ["21:00"], None)),
    )
    say(world, patient, "nahi")

    assert reminders(world) == []
    assert len(world.texts_to(PATIENT)) == 2


def test_a_reminder_request_without_a_medicine_and_time_asks_for_them(
    world: ReminderWorld,
):
    _, patient = setup(world)

    say(world, patient, "mujhe yaad dilana", extracted())

    [ask] = world.texts_to(PATIENT)
    assert "8 baje" in ask  # an example of what to send
    assert reminders(world) == []


@pytest.mark.parametrize("patient_wrote_today", [True, False])
def test_a_due_reminder_is_sent_and_the_next_one_scheduled(
    world: ReminderWorld, patient_wrote_today: bool
):
    _, patient = setup(world)
    reminder = saved_reminder(world, patient, times=("08:00", "20:00"))
    make_due(world, reminder)
    if not patient_wrote_today:
        with world.engine.begin() as conn:
            conn.execute(
                text("UPDATE messages SET created_at = now() - interval '25 hours'")
            )

    assert world.reminders.send_due()
    assert not world.reminders.send_due()  # nothing else is due

    if patient_wrote_today:
        [reminder_text] = world.texts_to(PATIENT)
        assert "Metformin" in reminder_text
    else:
        [(to, name, params)] = world.templates
        assert (to, name) == (PATIENT, "medicine_reminder")
        assert params[0] == "Metformin"
    assert doses(world) == [(reminder, "pending")]
    [row] = reminders(world)
    assert row.next_due_at > datetime.now(UTC)


def test_a_taken_reply_marks_the_dose_taken(world: ReminderWorld):
    _, patient = setup(world)
    reminder = saved_reminder(world, patient)
    make_due(world, reminder)
    world.reminders.send_due()
    world.sent.clear()

    message = say(world, patient, "maine dawai le li")

    assert doses(world) == [(reminder, "taken")]
    assert len(world.texts_to(PATIENT)) == 1
    assert world.agent_calls == []  # it never reached the Q&A agent
    assert world.message(message).status == "done"


def test_without_a_pending_dose_le_li_is_an_ordinary_message(world: ReminderWorld):
    _, patient = setup(world)

    say(world, patient, "le li")
    say(world, patient, "haan")  # no read-back is waiting for it

    assert [agent for agent, _ in world.agent_calls] == ["qa", "qa"]


def test_a_message_that_says_more_than_taken_stays_a_question(world: ReminderWorld):
    _, patient = setup(world)
    reminder = saved_reminder(world, patient)
    make_due(world, reminder)
    world.reminders.send_due()

    say(world, patient, "dawai li thi par pet mein dard ho raha hai")

    assert doses(world) == [(reminder, "pending")]
    assert [agent for agent, _ in world.agent_calls] == ["qa"]


def test_three_missed_doses_in_a_row_alert_the_asha_once(world: ReminderWorld):
    asha, patient = setup(world)
    world.receive(asha, body="namaste")
    world.worker.work_one()
    reminder = saved_reminder(world, patient)
    for _ in range(5):
        make_due(world, reminder)
        world.reminders.send_due()

    # Each dose is missed once the next is sent; the fifth is still pending.
    assert [d.status for d in doses(world)] == ["missed"] * 4 + ["pending"]
    assert [c.cause for c in world.cases()] == ["missed_doses"]
    [alert] = [body for to, body in world.sent if to == ASHA and "Case" in body]
    assert "Metformin" in alert and "3" in alert
    # The patient asked nothing, so gets reminders only, no "we told your ASHA".
    assert all("Metformin" in body for body in world.texts_to(PATIENT))


def test_a_taken_dose_breaks_the_streak(world: ReminderWorld):
    _, patient = setup(world)
    reminder = saved_reminder(world, patient)
    for n in range(5):
        make_due(world, reminder)
        world.reminders.send_due()
        if n == 1:
            say(world, patient, "taken")

    assert [d.status for d in doses(world)] == [
        "missed",
        "taken",
        "missed",
        "missed",
        "pending",
    ]
    assert world.cases() == []


def test_the_last_dose_of_a_course_ends_it(world: ReminderWorld):
    _, patient = setup(world)
    reminder = saved_reminder(world, patient, days=1)
    with world.engine.begin() as conn:  # a one-day course that was yesterday
        conn.execute(
            text(
                "UPDATE reminders SET starts_on = current_date - 1, "
                "ends_on = current_date - 1"
            )
        )
    make_due(world, reminder, ago=timedelta(hours=14))

    world.reminders.send_due()

    [row] = reminders(world)
    assert row.next_due_at is None


def test_a_confirmed_prescription_becomes_reminders_the_same_way(world: ReminderWorld):
    asha, patient = setup(world)
    with world.engine.begin() as conn:
        number = conn.execute(
            text(
                "INSERT INTO cases (cause, patient_id, asha_id, details, prescription,"
                " status, asha_reply, answered_at) VALUES ('prescription_confirmation',"
                " :p, :a, 'list', 'Dolo 650: 1-0-1, 5 days\nAmlodipine 5: 0-0-1',"
                " 'answered', 'ok', now()) RETURNING number"
            ),
            {"p": patient, "a": asha},
        ).scalar_one()
    assert number

    say(
        world,
        patient,
        "haan, subah 8 baje aur raat 9 baje yaad dilana",
        extracted(
            ("Dolo 650", ["08:00", "21:00"], 5), ("Amlodipine 5", ["21:00"], None)
        ),
    )
    # The LLM sees the medicines the ASHA confirmed, to name them.
    assert "Amlodipine 5: 0-0-1" in world.chat_requests[0]["messages"][0]["content"]
    say(world, patient, "yes")

    assert [r.medicine for r in reminders(world)] == ["Dolo 650", "Amlodipine 5"]
    assert all(r.next_due_at is not None for r in reminders(world))


def test_a_course_ends_after_its_last_day_in_india_time():
    course = ([time(8), time(20)], date(2026, 1, 1), date(2026, 1, 2))

    def at(day: int, hour: int, minute: int = 0) -> datetime:
        return datetime(2026, 1, day, hour, minute, tzinfo=IST)

    assert next_due(*course, at(1, 7)) == at(1, 8)
    assert next_due(*course, at(1, 8)) == at(1, 20)  # strictly after
    assert next_due(*course, at(1, 21)) == at(2, 8)
    assert next_due(*course, at(2, 20, 30)) is None
    due = next_due(*course, at(2, 1))
    assert due is not None and due.tzinfo == UTC  # stored in UTC


def test_a_reply_without_words_neither_saves_nor_discards(world: ReminderWorld):
    _, patient = setup(world)
    say(
        world,
        patient,
        "metformin yaad dilana",
        extracted(("Metformin", ["08:00"], None)),
    )

    say(world, patient, "👍")

    [draft] = reminders(world)
    assert draft.confirmed_at is None
    assert [agent for agent, _ in world.agent_calls] == ["qa"]


def test_a_course_asked_for_after_todays_last_time_starts_tomorrow(
    world: ReminderWorld,
):
    _, patient = setup(world)
    say(world, patient, "raat 12 baje yaad dilana", extracted(("ORS", ["00:01"], 1)))

    say(world, patient, "haan")

    [saved] = reminders(world)
    assert saved.next_due_at is not None  # one reminder, the day after asking
    assert saved.starts_on == saved.ends_on == saved.next_due_at.astimezone(IST).date()


def test_a_worker_back_after_days_sends_one_late_reminder_not_a_burst(
    world: ReminderWorld,
):
    _, patient = setup(world)
    reminder = saved_reminder(world, patient, times=("08:00", "20:00"))
    with world.engine.begin() as conn:  # a course running for a week
        conn.execute(text("UPDATE reminders SET starts_on = current_date - 7"))
    make_due(world, reminder, ago=timedelta(days=3))

    assert world.reminders.send_due()
    assert not world.reminders.send_due()

    assert len(doses(world)) == 1
    [row] = reminders(world)
    assert row.next_due_at > datetime.now(UTC)
