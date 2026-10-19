"""keep a prescription's medicines and its photo on its case

Revision ID: 0006
Revises: 0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("cases", sa.Column("prescription", sa.Text(), nullable=True))
    op.add_column("cases", sa.Column("photo_media_id", sa.Text(), nullable=True))
    op.create_check_constraint(
        op.f("ck_cases_prescription_has_medicines"),
        "cases",
        "(cause = 'prescription_confirmation') = (prescription IS NOT NULL)",
    )
    op.create_check_constraint(
        op.f("ck_cases_only_prescriptions_have_a_photo"),
        "cases",
        "photo_media_id IS NULL OR prescription IS NOT NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_cases_only_prescriptions_have_a_photo"), "cases", type_="check"
    )
    op.drop_constraint(
        op.f("ck_cases_prescription_has_medicines"), "cases", type_="check"
    )
    op.drop_column("cases", "photo_media_id")
    op.drop_column("cases", "prescription")
