"""add disposition to inventory items

Revision ID: b7e2f1a3c4d5
Revises: a409d81e2f55
Create Date: 2026-04-05 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7e2f1a3c4d5'
down_revision: Union[str, None] = 'a409d81e2f55'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('inventory_items', sa.Column('disposition', sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column('inventory_items', 'disposition')
