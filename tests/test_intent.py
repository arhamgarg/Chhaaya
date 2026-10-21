import json

import httpx2
import pytest

from chhaaya.intent import evaluate, sarvam_zero_shot
from chhaaya.sarvam import SarvamClient


def test_each_system_is_scored_per_language_with_macro_f1_over_the_four_labels():
    rows = [
        {"text": text, "label": label, "language": language}
        for text, label, language in [
            ("bukhar mein kya khayein", "question", "hinglish"),
            ("shukriya didi", "smalltalk", "hinglish"),
            ("roz 8 baje yaad dilana", "reminder_setup", "hinglish"),
            ("le li", "reminder_reply", "hinglish"),
            ("what should i eat in fever", "question", "en"),
            ("thank you so much", "smalltalk", "en"),
            ("remind me at 8", "reminder_setup", "en"),
            ("taken", "reminder_reply", "en"),
        ]
    ]
    # Sarvam-105B's replies, in row order. A reply that is not a label, or none
    # because reasoning used up the tokens (None), counts as the router's
    # default, question.
    replies = iter(
        ["question", "I am not sure", "reminder_setup", "`reminder_reply`"]
        + [None, "smalltalk", "reminder_reply", "reminder_reply"]
    )
    prompts: list[str] = []

    def chat(request: httpx2.Request) -> httpx2.Response:
        prompts.append(json.loads(request.read())["messages"][0]["content"])
        reply = next(replies)
        message = {"role": "assistant", "content": reply or ""}
        stop = "stop" if reply else "length"
        choice = {"index": 0, "message": message, "finish_reason": stop}
        return httpx2.Response(200, json={"choices": [choice]})

    sarvam = SarvamClient("key", httpx2.Client(transport=httpx2.MockTransport(chat)))

    results = evaluate(rows, {"Sarvam-105B": sarvam_zero_shot(sarvam)})

    [result] = results
    assert result.system == "Sarvam-105B"
    # Hinglish: question F1 2/3 (one smalltalk read as a question), smalltalk 0,
    # the reminder labels 1.
    assert result.macro_f1["hinglish"] == pytest.approx((2 / 3 + 0 + 1 + 1) / 4)
    # English: setup F1 0 and reply F1 2/3 (one setup read as a reply).
    assert result.macro_f1["en"] == pytest.approx((1 + 1 + 0 + 2 / 3) / 4)
    assert result.invalid == 2
    assert len(prompts) == len(rows)
    assert all("reminder_reply" in prompt for prompt in prompts)
