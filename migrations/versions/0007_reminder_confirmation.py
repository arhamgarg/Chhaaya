"""keep a reminder unsent until the patient confirms it

Revision ID: 0007
Revises: 0006
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "reminders",
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_reminders_only_confirmed_reminders_are_due"),
        "reminders",
        "confirmed_at IS NOT NULL OR next_due_at IS NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_reminders_only_confirmed_reminders_are_due"),
        "reminders",
        type_="check",
    )
    op.drop_column("reminders", "confirmed_at")
