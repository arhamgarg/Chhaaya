"""Embeds the knowledge corpus into `kb_chunks` and searches it by cosine similarity."""

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sqlalchemy import Engine, create_engine, delete, insert, select

from chhaaya.corpus import CHUNKS
from chhaaya.db import KbChunk, sqlalchemy_url
from chhaaya.settings import Settings

MODEL = "BAAI/bge-m3"

Embed = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True)
class Hit:
    id: int
    text: str
    title: str
    publisher: str | None
    year: int | None
    page: int | None
    url: str | None
    source_type: str
    score: float  # cosine similarity to the query; 1 means the same direction


def bge_m3() -> Embed:
    """Load BGE-M3 on the CPU; the first call downloads it (about 2 GB)."""
    model = SentenceTransformer(MODEL, device="cpu")
    return lambda texts: model.encode(list(texts), batch_size=16).tolist()


class KnowledgeBase:
    def __init__(self, engine: Engine, embed: Embed) -> None:
        self.engine = engine
        self.embed = embed

    def load(self, path: Path = CHUNKS) -> int:
        """Replace every guideline chunk with those in `path`; ASHA answers stay."""
        lines = path.read_text(encoding="utf-8").splitlines()
        chunks = [json.loads(line) for line in lines]
        if not chunks:
            raise ValueError(f"{path} has no chunks")
        embeddings = self.embed([c["text"] for c in chunks])
        rows = [
            {
                "text": c["text"],
                "title": c["title"],
                "publisher": c["publisher"],
                "year": c["year"],
                "page": c["page"],
                "url": c["url"],
                "source_type": "guideline",
                "embedding": embedding,
            }
            for c, embedding in zip(chunks, embeddings, strict=True)
        ]
        with self.engine.begin() as conn:
            conn.execute(delete(KbChunk).where(KbChunk.source_type == "guideline"))
            conn.execute(insert(KbChunk), rows)
        return len(rows)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """The k chunks closest to `query`, best first."""
        distance = KbChunk.embedding.cosine_distance(self.embed([query])[0])
        statement = (
            select(KbChunk, distance.label("distance")).order_by(distance).limit(k)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(statement).all()
        return [
            Hit(
                id=row.id,
                text=row.text,
                title=row.title,
                publisher=row.publisher,
                year=row.year,
                page=row.page,
                url=row.url,
                source_type=row.source_type,
                score=1 - row.distance,
            )
            for row in rows
        ]


def main() -> None:
    engine = create_engine(sqlalchemy_url(Settings().database_url))
    loaded = KnowledgeBase(engine, bge_m3()).load()
    print(f"{loaded} chunks from {CHUNKS} -> kb_chunks", file=sys.stderr)


if __name__ == "__main__":
    main()
