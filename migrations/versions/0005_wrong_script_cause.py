"""escalate answers in the wrong script

Revision ID: 0005
Revises: 0004
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CAUSES = (
    "'danger_sign', 'weak_evidence', 'citation_failed', 'critical_lab_value', "
    "'prescription_confirmation', 'missed_doses'"
)


def upgrade() -> None:
    op.drop_constraint(op.f("ck_cases_cause"), "cases", type_="check")
    op.create_check_constraint(
        op.f("ck_cases_cause"), "cases", f"cause IN ({CAUSES}, 'wrong_script')"
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_cases_cause"), "cases", type_="check")
    op.create_check_constraint(op.f("ck_cases_cause"), "cases", f"cause IN ({CAUSES})")
