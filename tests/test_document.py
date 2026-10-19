import json

import pytest
from sqlalchemy import Engine, text
from test_worker import World

from chhaaya.cases import NOTICE
from chhaaya.document import DocumentAgent
from chhaaya.lexicon import Entry, Lexicon

ASHA = "919999999999"
PATIENT = "911111111111"
LEXICON = Lexicon(
    [
        Entry("Dolo 650 mg oral tablet", "Paracetamol 650 mg oral tablet", "", "c1"),
        Entry("Metformin", "Metformin", "Tablet 500 mg", "NLEM 2022 17.5.1"),
        Entry("Amlodipine", "Amlodipine", "Tablet 5 mg", "NLEM 2022 12.4.1"),
    ]
)

CBC = """\
# City Diagnostics
Patient: Sunita Devi   Age/Sex: 34 Y / F

| Test | Result | Unit | Reference range |
|---|---|---|---|
| Haemoglobin | 6.8 L | g/dL | 12.0 - 15.0 |
| Mean corpuscular haemoglobin (MCH) | 24 | pg | 27 - 32 |
| Platelet count | 250000 | /cumm | 150000 - 450000 |
| TSH | 3.1 | uIU/mL | |
"""
CBC_TESTS = [
    {"name": "Haemoglobin", "value": "6.8", "unit": "g/dL", "range": "12.0 - 15.0"},
    {
        "name": "Mean corpuscular haemoglobin (MCH)",
        "value": "24",
        "unit": "pg",
        "range": "27 - 32",
    },
    {
        "name": "Platelet count",
        "value": "250000",
        "unit": "/cumm",
        "range": "150000 - 450000",
    },
    {"name": "TSH", "value": "3.1", "unit": "uIU/mL", "range": None},
]
RX = """\
Dr. A. Sharma, MBBS
Rx
1. Tab Dolo 650   1-0-1 x 5 days
2. Tab Metfornin 500   0-0-1 x 1 month
"""
RX_MEDICINES = [
    {"name": "Dolo", "strength": "650", "frequency": "1-0-1", "duration": "5 days"},
    {
        "name": "Metfornin",
        "strength": "500",
        "frequency": "0-0-1",
        "duration": "1 month",
    },
]


@pytest.fixture
def world(engine: Engine) -> World:
    world = World(engine)
    cases = world.cases_
    agent = DocumentAgent(
        world.worker.sarvam, world.worker.whatsapp, cases, lambda: LEXICON
    )
    world.agents["document"] = agent.read
    world.agents["escalation"] = cases.answer
    return world


def setup(
    world: World, language=("en", "Latn"), asha_wrote_today=True
) -> tuple[int, int]:
    asha = world.add_user(ASHA, "asha")
    patient = world.add_user(PATIENT, asha=asha)
    with world.engine.begin() as conn:
        conn.execute(
            text("UPDATE users SET language = :l, script = :s WHERE id = :id"),
            {"l": language[0], "s": language[1], "id": patient},
        )
    if asha_wrote_today:
        world.receive(asha, body="namaste")
        world.worker.work_one()  # no case number: the ASHA gets a usage hint
    world.sent.clear()
    return asha, patient


def extracted(kind: str, tests=(), medicines=(), age: float | None = None) -> str:
    return json.dumps(
        {
            "kind": kind,
            "patient_age_years": age,
            "tests": list(tests),
            "medicines": list(medicines),
        }
    )


def send_photo(world: World, patient: int, ocr: str | None, *replies: str) -> int:
    world.ocr = ocr
    world.chat_replies = list(replies)
    message = world.receive(patient, kind="image")
    world.worker.work_one()
    return message


def case_details(world: World):
    with world.engine.connect() as conn:
        return conn.execute(text("SELECT number, details, status FROM cases")).all()


def test_a_lab_report_is_compared_with_its_ranges_and_a_critical_value_escalates(
    world: World,
):
    _, patient = setup(world)

    message = send_photo(
        world, patient, CBC, extracted("lab_report", CBC_TESTS, age=34)
    )

    # The case opens first, so the ASHA is alerted even if the explanation fails.
    [(number, details, _)] = case_details(world)
    assert [c.cause for c in world.cases()] == ["critical_lab_value"]
    assert "Haemoglobin 6.8 g/dL is below the critical threshold of 8 g/dL" in details
    assert "MCH" not in details  # 24 pg is a different test, never haemoglobin
    alert, *_ = (body for _, body in world.sent)
    assert alert.startswith(f"Case {number}: critical lab value")
    [notice, explanation] = world.texts_to(PATIENT)
    assert notice in NOTICE.values()
    lines = explanation.splitlines()
    assert any(line.startswith("Haemoglobin 6.8 g/dL: below") for line in lines)
    assert any(
        line.startswith("Mean corpuscular haemoglobin (MCH) 24 pg: below")
        for line in lines
    )
    assert any(line.startswith("Platelet count 250000 /cumm: inside") for line in lines)
    assert any(line.startswith("TSH 3.1 uIU/mL: no range") for line in lines)
    assert world.message(message).status == "done"
    # The OCR text is what the LLM structures, and nothing else.
    assert world.chat_requests[0]["messages"][1]["content"] == CBC


@pytest.mark.parametrize(
    ("age", "critical"),
    [
        (3, False),  # under 5 the severe cut-off is 7 g/dL
        (None, True),  # no age printed: the highest cut-off, 8 g/dL, applies
    ],
)
def test_a_report_without_printed_ranges_is_still_checked_for_critical_values(
    world: World, age: float | None, critical: bool
):
    _, patient = setup(world, language=("hi", "Latn"))
    report = "Govt PHC Lab\nHb : 7.5 gm%\nBlood sugar (random) : 96 mg/dl\n"
    tests = [
        {"name": "Hb", "value": "7.5", "unit": "gm%", "range": None},
        {"name": "Blood sugar (random)", "value": "96", "unit": "mg/dl", "range": None},
    ]

    send_photo(world, patient, report, extracted("lab_report", tests, age=age))

    assert [c.cause for c in world.cases()] == (
        ["critical_lab_value"] if critical else []
    )
    explanation = world.texts_to(PATIENT)[-1]
    assert "Hb 7.5 gm%" in explanation and "Blood sugar (random) 96" in explanation
    assert explanation.count("range chhapi nahi") == 2


def test_a_critical_test_in_a_unit_we_cannot_check_goes_to_the_asha(world: World):
    _, patient = setup(world)
    report = "Haemoglobin 75 g/L (120 - 150)"
    tests = [
        {"name": "Haemoglobin", "value": "75", "unit": "g/L", "range": "120 - 150"}
    ]

    send_photo(world, patient, report, extracted("lab_report", tests))

    [(_, details, _)] = case_details(world)
    assert "g/L" in details and "could not check" in details
    assert "Haemoglobin 75 g/L: below" in world.texts_to(PATIENT)[-1]


@pytest.mark.parametrize(
    ("ocr", "reply", "status"),
    [
        # A blurry photo: Sarvam Vision reads fragments, the LLM calls it unreadable.
        ("H@em0g ~~ 1?.? g/d", extracted("unreadable"), "done"),
        # A value the OCR text does not contain was made up, not read.
        (CBC, extracted("lab_report", [{**CBC_TESTS[0], "value": "11.8"}]), "failed"),
        (CBC, "Here is the JSON you asked for: {}", "failed"),
        (CBC, extracted("lab_report"), "failed"),  # a report with no tests
        (None, None, "failed"),  # the Sarvam Vision job fails
    ],
    ids=["blurry", "invented-value", "not-json", "no-tests", "ocr-failed"],
)
def test_a_photo_that_cannot_be_read_reliably_asks_for_a_clearer_one(
    world: World, ocr: str | None, reply: str | None, status: str
):
    _, patient = setup(world)

    message = send_photo(world, patient, ocr, *([reply] if reply else []))

    [answer] = world.texts_to(PATIENT)
    assert "clear photo" in answer
    assert "6.8" not in answer and "11.8" not in answer
    assert world.cases() == []
    assert world.message(message).status == status


def test_a_prescription_reaches_the_patient_only_after_the_asha_confirms_it(
    world: World,
):
    asha, patient = setup(world, language=("hi", "Latn"))
    reply = extracted("prescription", medicines=RX_MEDICINES)

    send_photo(world, patient, RX, f"```json\n{reply}\n```")  # fenced, as LLMs do

    [(number, details, _)] = case_details(world)
    # The ASHA gets the photo and the list read from it, with the lexicon's view.
    assert world.images == [(ASHA, f"Case {number}: the prescription photo")]
    [alert] = world.texts_to(ASHA)
    assert "Dolo 650: 1-0-1, 5 days" in alert and "Paracetamol 650 mg" in alert
    assert "Metfornin 500: 0-0-1, 1 month" in alert and "closest: Metformin" in alert
    # The patient hears nothing about the medicines yet.
    [notice] = world.texts_to(PATIENT)
    assert notice in NOTICE.values()

    world.sent.clear()
    world.receive(asha, body=f"{number} OK")
    world.worker.work_one()

    [confirmed] = world.texts_to(PATIENT)
    assert "Dolo 650: 1-0-1, 5 days" in confirmed
    assert "Metfornin 500: 0-0-1, 1 month" in confirmed  # as written, as confirmed
    assert "Paracetamol" not in confirmed and "closest" not in confirmed
    assert "reminder" in confirmed
    assert case_details(world)[0].status == "answered"


def test_an_asha_outside_the_window_sees_the_photo_before_her_answer_counts(
    world: World,
):
    asha, patient = setup(world, asha_wrote_today=False)

    send_photo(world, patient, RX, extracted("prescription", medicines=RX_MEDICINES))

    [(number, _, _)] = case_details(world)
    [(to, name, _)] = world.templates
    assert (to, name) == (ASHA, "asha_case_alert")
    assert world.images == []  # Meta refuses an image outside the window

    world.sent.clear()
    world.receive(asha, body=f"{number} ok")  # she has not seen the photo yet
    world.worker.work_one()

    assert world.images == [(ASHA, f"Case {number}: the prescription photo")]
    [shown] = world.texts_to(ASHA)
    assert "Dolo 650: 1-0-1, 5 days" in shown
    assert world.texts_to(PATIENT) == []
    assert case_details(world)[0].status == "open"

    world.receive(asha, body=f"{number} Dolo 650 1-0-1 for 3 days only")
    world.worker.work_one()

    [corrected] = world.texts_to(PATIENT)
    assert "Dolo 650 1-0-1 for 3 days only" in corrected
    assert "5 days" not in corrected and "Metfornin" not in corrected
    assert case_details(world)[0].status == "answered"


def test_a_photo_of_something_else_is_not_read_as_a_document(world: World):
    _, patient = setup(world)

    send_photo(world, patient, "A child smiling in a garden", extracted("other"))

    [answer] = world.texts_to(PATIENT)
    assert "lab reports and prescriptions" in answer
    assert world.cases() == []


def test_a_long_prescription_reaches_the_asha_whole(world: World):
    _, patient = setup(world)
    medicines = [
        {
            "name": f"Medicine{n}",
            "strength": "500",
            "frequency": "1-0-1",
            "duration": "5 days",
        }
        for n in range(12)
    ]
    ocr = "\n".join(f"Tab Medicine{n} 500 1-0-1 x 5 days" for n in range(12))

    send_photo(world, patient, ocr, extracted("prescription", medicines=medicines))

    [alert] = world.texts_to(ASHA)
    assert "Medicine11 500: 1-0-1, 5 days" in alert  # the ASHA confirms all of it


@pytest.mark.parametrize(
    ("name", "unit", "critical"),
    [
        ("Hb (Sahli)", "g/dL", True),
        ("Hb level", "g/dL", True),
        ("Haemoglobin estimation", "gm/dl", True),
        ("HbA1c", "%", False),
        ("Glycated haemoglobin", "%", False),
        ("Mean corpuscular haemoglobin concentration (MCHC)", "g/dL", False),
    ],
)
def test_every_haemoglobin_name_is_checked_and_other_tests_are_not(
    world: World, name: str, unit: str, critical: bool
):
    _, patient = setup(world)
    tests = [{"name": name, "value": "5.5", "unit": unit, "range": None}]

    send_photo(world, patient, f"{name} 5.5 {unit}", extracted("lab_report", tests))

    assert [c.cause for c in world.cases()] == (
        ["critical_lab_value"] if critical else []
    )


@pytest.mark.parametrize(
    "answer", ["ok 👍", "ok?", "yes", "haan ji", "Ok sahi hai", "ठीक है।", "हाँ जी"]
)
def test_any_way_an_asha_says_yes_confirms_the_prescription(world: World, answer: str):
    asha, patient = setup(world)
    send_photo(world, patient, RX, extracted("prescription", medicines=RX_MEDICINES))
    [(number, _, _)] = case_details(world)
    world.sent.clear()

    world.receive(asha, body=f"{number} {answer}")
    world.worker.work_one()

    [confirmed] = world.texts_to(PATIENT)
    assert "Dolo 650: 1-0-1, 5 days" in confirmed
    assert case_details(world)[0].status == "answered"


def test_a_reply_that_neither_confirms_nor_lists_medicines_keeps_the_case_open(
    world: World,
):
    asha, patient = setup(world)
    send_photo(world, patient, RX, extracted("prescription", medicines=RX_MEDICINES))
    [(number, _, _)] = case_details(world)
    world.sent.clear()

    world.receive(asha, body=f"{number} dekh liya, baad mein batati hoon")
    world.worker.work_one()

    assert world.texts_to(PATIENT) == []
    [hint] = world.texts_to(ASHA)
    assert f"{number} ok" in hint
    assert case_details(world)[0].status == "open"


def test_a_long_report_for_a_patient_of_unknown_language_still_arrives(world: World):
    _, patient = setup(world, language=(None, None))
    tests = [
        {
            "name": f"Test number {n}",
            "value": f"{n}.5",
            "unit": "mg/dL",
            "range": f"{n} - {n + 1}",
        }
        for n in range(40)
    ]
    ocr = "\n".join(f"{t['name']} {t['value']} {t['unit']} {t['range']}" for t in tests)

    message = send_photo(world, patient, ocr, extracted("lab_report", tests))

    texts = world.texts_to(PATIENT)
    assert all(len(body) <= 4096 for body in texts)
    assert "Test number 39 39.5 mg/dL" in "".join(texts)
    assert world.message(message).status == "done"
