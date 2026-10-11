import json
import re
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, insert, select

from chhaaya.corpus import CHUNKS, read_sources
from chhaaya.db import EMBEDDING_DIMENSIONS, KbChunk
from chhaaya.retrieval import KnowledgeBase, bge_m3

# Stands in for BGE-M3, which is a 2 GB download: each known word is one axis.
WORDS = ["ors", "zinc", "vaccine", "measles", "fever", "pregnancy", "iron", "salt"]


def fake_embed(texts: Sequence[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        vector = [0.0] * EMBEDDING_DIMENSIONS
        for word in re.findall(r"\w+", text.lower()):
            if word in WORDS:
                vector[WORDS.index(word)] += 1
        vector[-1] = 0.1  # never the zero vector, whose cosine is undefined
        vectors.append(vector)
    return vectors


def chunk(n: int, text: str, source: str = "idcf_guidelines") -> dict:
    return {
        "id": f"{source}:{n}",
        "source_id": source,
        "title": f"Title of {source}",
        "publisher": "MoHFW",
        "year": 2019,
        "url": f"https://example.org/{source}.pdf",
        "page": n + 10,
        "section": "Section",
        "text": text,
        "tokens": len(text.split()),
    }


def write(path: Path, chunks: list[dict]) -> Path:
    path.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    return path


def guideline_texts(engine: Engine) -> list[str]:
    with engine.connect() as conn:
        return sorted(
            conn.scalars(select(KbChunk.text).where(KbChunk.source_type == "guideline"))
        )


def test_reloading_replaces_guidelines_and_keeps_health_worker_answers(
    engine: Engine, tmp_path: Path
):
    with engine.begin() as conn:
        conn.execute(
            insert(KbChunk).values(
                text="ASHA answer: give ORS after every loose stool",
                title="Answered earlier by a health worker",
                source_type="health_worker_answer",
                embedding=fake_embed(["ors"])[0],
            )
        )
    kb = KnowledgeBase(engine, fake_embed)
    first = write(
        tmp_path / "first.jsonl",
        [chunk(1, "ORS for diarrhoea"), chunk(2, "zinc for 14 days")],
    )
    assert kb.load(first) == 2
    assert kb.load(first) == 2
    assert guideline_texts(engine) == ["ORS for diarrhoea", "zinc for 14 days"]

    second = write(tmp_path / "second.jsonl", [chunk(1, "ORS, revised edition")])
    kb.load(second)
    assert guideline_texts(engine) == ["ORS, revised edition"]
    with engine.connect() as conn:
        answers = conn.scalar(
            select(func.count()).where(KbChunk.source_type == "health_worker_answer")
        )
    assert answers == 1


def test_an_empty_corpus_file_fails_and_leaves_the_knowledge_base_intact(
    engine: Engine, tmp_path: Path
):
    kb = KnowledgeBase(engine, fake_embed)
    kb.load(write(tmp_path / "chunks.jsonl", [chunk(1, "ORS for diarrhoea")]))
    with pytest.raises(ValueError, match="no chunks"):
        kb.load(write(tmp_path / "empty.jsonl", []))
    assert guideline_texts(engine) == ["ORS for diarrhoea"]


def test_search_ranks_by_cosine_similarity_with_scores_and_metadata(
    engine: Engine, tmp_path: Path
):
    kb = KnowledgeBase(engine, fake_embed)
    kb.load(
        write(
            tmp_path / "chunks.jsonl",
            [
                chunk(1, "measles vaccine at nine months", "immunization_handbook"),
                chunk(2, "ORS and zinc for diarrhoea"),
                chunk(3, "ORS ORS ORS after each stool"),
                chunk(4, "iron tablets in pregnancy", "sba_guidelines"),
                chunk(5, "less salt for high blood pressure", "np_ncd_guidelines"),
                chunk(6, "fever in children", "icmr_stw_fever_children"),
                chunk(7, "zinc zinc zinc"),
            ],
        )
    )

    hits = kb.search("how much ORS for my child?", k=3)

    assert [h.text for h in hits][:2] == [
        "ORS ORS ORS after each stool",
        "ORS and zinc for diarrhoea",
    ]
    assert len(hits) == 3
    assert hits[0].score == pytest.approx(1.0, abs=0.01)
    assert hits[0].score > hits[1].score > hits[2].score
    top = hits[0]
    assert (top.title, top.publisher, top.year, top.page, top.url, top.source_type) == (
        "Title of idcf_guidelines",
        "MoHFW",
        2019,
        13,
        "https://example.org/idcf_guidelines.pdf",
        "guideline",
    )
    assert len(kb.search("vaccine")) == 5


# Hand-written patient questions, each with the documents that answer it.
QUERIES = [
    (
        "en",
        "How much ORS and zinc should a child with loose motions get?",
        {"idcf_guidelines", "icmr_stw_diarrhoea_children", "hbyc_asha_handbook"},
    ),
    (
        "en",
        "When is the measles rubella vaccine given to a baby?",
        {"immunization_handbook"},
    ),
    (
        "en",
        "Who at home should get medicine to prevent TB when a family member has TB?",
        {"tb_preventive_treatment", "icmr_stw_paediatric_tb"},
    ),
    (
        "en",
        "How often should people over 30 get their blood pressure checked?",
        {"np_ncd_guidelines", "icmr_stw_hypertension"},
    ),
    (
        "hi",
        "गर्भावस्था में कितनी बार जांच करानी चाहिए?",
        {"sba_guidelines", "asha_module_6"},
    ),
    (
        "hi",
        "नवजात शिशु को ठंड से कैसे बचाएं?",
        {"asha_module_6", "asha_module_7", "hbyc_asha_handbook"},
    ),
    (
        "hi",
        "बच्चे को दस्त में ओआरएस कैसे बनाकर दें?",
        {"idcf_guidelines", "icmr_stw_diarrhoea_children", "hbyc_asha_handbook"},
    ),
    (
        "hi",
        "डायबिटीज़ में खाने का ध्यान कैसे रखें?",
        {"icmr_stw_diabetes_type_2", "np_ncd_guidelines"},
    ),
    (
        "hinglish",
        "baby ko pehle 6 mahine sirf maa ka doodh dena chahiye kya?",
        {"hbyc_asha_handbook", "asha_module_6", "asha_module_7"},
    ),
    (
        "hinglish",
        "pregnancy mein iron ki goli kab se leni hai?",
        {"sba_guidelines", "asha_module_6"},
    ),
    (
        "hinglish",
        "bachche ko tez bukhar hai, kab doctor ke paas le jaana chahiye?",
        {"icmr_stw_fever_children", "hbyc_asha_handbook", "asha_module_7"},
    ),
    (
        "hinglish",
        "BP high rehta hai, namak kitna kam karna chahiye?",
        {"icmr_stw_hypertension", "np_ncd_guidelines"},
    ),
]


@pytest.mark.retrieval
def test_hand_written_questions_find_their_document_in_the_top_5(engine: Engine):
    """Needs data/chunks.jsonl (`python -m chhaaya.corpus`) and the BGE-M3 model."""
    source_of = {s.url: s.id for s in read_sources()}
    kb = KnowledgeBase(engine, bge_m3())
    started = time.perf_counter()
    loaded = kb.load(CHUNKS)
    print(f"\nembedded {loaded} chunks in {time.perf_counter() - started:.0f} s")

    misses, latencies = [], []
    for language, query, expected in QUERIES:
        started = time.perf_counter()
        hits = kb.search(query, k=5)
        latencies.append(time.perf_counter() - started)
        found = [source_of[h.url] for h in hits if h.url]
        print(f"{language:8} {hits[0].score:.3f} {query} -> {found}")
        if not expected & set(found):
            misses.append((language, query, found))
    for language in ("en", "hi", "hinglish"):
        asked = [q for q in QUERIES if q[0] == language]
        missed = [m for m in misses if m[0] == language]
        print(f"hit@5 {language}: {len(asked) - len(missed)}/{len(asked)}")
    print(
        f"search latency over {len(latencies)} queries: median "
        f"{statistics.median(latencies) * 1000:.0f} ms, max "
        f"{max(latencies) * 1000:.0f} ms"
    )
    assert misses == []
