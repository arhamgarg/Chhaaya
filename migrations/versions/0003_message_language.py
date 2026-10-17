"""store the language of a transcribed voice note on the message

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("language", sa.Text(), nullable=True))
    op.create_check_constraint(
        op.f("ck_messages_language"), "messages", "language IN ('hi', 'en')"
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_messages_language"), "messages")
    op.drop_column("messages", "language")
