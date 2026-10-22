"""A saved CPU classifier routes signed messages, with real Postgres and agents."""

import json
from pathlib import Path

import pytest
import torch
from test_qa import PATIENT, add_chunk, setup
from test_retrieval import fake_embed
from test_webhook import client as client
from test_webhook import delivery, post
from test_worker import World
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from chhaaya.intent import Classifier
from chhaaya.qa import QaAgent
from chhaaya.reminders import Reminders
from chhaaya.retrieval import KnowledgeBase


def saved_classifier(path: Path, label: str, strength: float) -> Classifier:
    labels = ["question", "reminder_setup", "reminder_reply", "smalltalk"]
    path.mkdir()
    vocab = path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ni\ntake\nmedicine\n")
    tokenizer = BertTokenizer(vocab=str(vocab))
    tokenizer.save_pretrained(path)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            id2label=dict(enumerate(labels)),
            label2id={name: i for i, name in enumerate(labels)},
        )
    )
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.bias[labels.index(label)] = strength
    model.save_pretrained(path)
    (path / "router.json").write_text(
        json.dumps({"max_length": 64, "confidence_threshold": 0.8})
    )
    return Classifier(path)


@pytest.mark.parametrize(
    ("label", "strength", "message", "route"),
    [
        (
            "reminder_setup",
            8.0,
            "My tablet needs an alarm at eight tonight",
            "reminder",
        ),
        ("question", 8.0, "Can a reminder help me remember how ORS works?", "qa"),
        ("reminder_setup", 0.1, "Please remind me how ORS works", "qa"),
        ("reminder_reply", 8.0, "I took ORS but what does it do?", "qa"),
        ("smalltalk", 8.0, "Good evening and thank you", "smalltalk"),
    ],
)
def test_saved_model_routes_messages_and_uncertainty_stays_a_question(
    client, engine, tmp_path, label, strength, message, route
):
    world = World(engine, speech_enabled=False)
    world.language = ("en-IN", "Latn")
    setup(world)
    worker = world.worker
    worker.classify = saved_classifier(tmp_path / "classifier", label, strength)
    worker.agents["reminder"] = Reminders(
        engine, worker.whatsapp, worker.ai, world.cases_
    ).handle
    guideline = add_chunk(world, "ORS replaces water and salts lost during diarrhoea")
    worker.agents["qa"] = QaAgent(
        KnowledgeBase(engine, fake_embed),
        worker.ai,
        worker.whatsapp,
        world.cases_,
        worker.escalate,
    ).answer
    world.chat_replies = (
        ['{"reminders":[{"medicine":"Metformin","times":["20:00"],"days":null}]}']
        if route == "reminder"
        else ["How ORS works", f"ORS replaces lost water and salts [{guideline}]."]
    )
    payload = delivery(
        {
            "from": PATIENT,
            "id": "wamid.classified",
            "type": "text",
            "text": {"body": message},
        }
    )
    assert post(client, payload).status_code == 200
    assert worker.work_one()
    [reply] = world.texts_to(PATIENT)
    with engine.connect() as conn:
        reminders = conn.exec_driver_sql("SELECT confirmed_at FROM reminders").all()
        status = conn.exec_driver_sql(
            "SELECT status FROM messages WHERE wa_message_id='wamid.classified'"
        ).scalar_one()
    assert status == "done"
    if route == "reminder":
        assert "Metformin" in reply and "8:00 pm" in reply
        assert "Reply yes" in reply
        assert reminders == [(None,)]
    elif route == "qa":
        assert "ORS replaces lost water and salts" in reply
        assert "Source: Intensified Diarrhoea Control Fortnight" in reply
        assert reminders == []
    else:
        assert "health question" in reply and "ASHA" in reply
        assert reminders == []
