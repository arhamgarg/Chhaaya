from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, make_url, text
from testcontainers.community.postgres import PostgresContainer

from chhaaya.db import sqlalchemy_url

REQUIRED_ENV = (
    "WHATSAPP_VERIFY_TOKEN",
    "WHATSAPP_APP_SECRET",
    "WHATSAPP_ACCESS_TOKEN",
    "WHATSAPP_PHONE_NUMBER_ID",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
)
ALEMBIC_INI = Path(__file__).parents[1] / "alembic.ini"


@pytest.fixture(scope="session")
def postgres() -> Iterator[str]:
    with PostgresContainer("pgvector/pgvector:pg16", driver=None) as container:
        yield container.get_connection_url()


@pytest.fixture
def database_url(postgres: str) -> Iterator[str]:
    """A fresh, empty database on the shared server, dropped after the test."""
    name = f"test_{uuid4().hex}"
    admin = create_engine(sqlalchemy_url(postgres), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    yield make_url(postgres).set(database=name).render_as_string(hide_password=False)
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()


def migrate(database_url: str, monkeypatch: pytest.MonkeyPatch, revision: str) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("POSTGRES_PASSWORD", "unused-by-migrations")
    for name in REQUIRED_ENV:
        monkeypatch.setenv(name, "unused-by-migrations")
    config = Config(ALEMBIC_INI)
    if revision == "base":
        command.downgrade(config, revision)
    else:
        command.upgrade(config, revision)


@pytest.fixture
def engine(database_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    migrate(database_url, monkeypatch, "head")
    engine = create_engine(sqlalchemy_url(database_url))
    yield engine
    engine.dispose()
