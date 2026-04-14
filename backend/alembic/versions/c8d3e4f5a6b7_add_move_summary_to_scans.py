"""add move_summary to scans

Revision ID: c8d3e4f5a6b7
Revises: b7e2f1a3c4d5
Create Date: 2026-04-05 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "c8d3e4f5a6b7"
down_revision = "b7e2f1a3c4d5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("scans", sa.Column("move_summary", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("scans", "move_summary")
