"""add status column to stations table

Revision ID: c713d9e8b83d
Revises: 
Create Date: 2026-10-28 11:50:09.589091

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c713d9e8b83d'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "stations",
        sa.Column(
            "status",
            sa.String(),
            nullable=False,
            server_default=sa.text("'inactive'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("stations", "status")
