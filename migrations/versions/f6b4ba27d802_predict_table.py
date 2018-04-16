"""predict table

Revision ID: f6b4ba27d802
Revises: 3854cb16db92
Create Date: 2018-04-16 11:52:52.046447

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f6b4ba27d802'
down_revision = '3854cb16db92'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('post', sa.Column('prediction', sa.Integer(), nullable=True))
    op.add_column('tweet', sa.Column('prediction', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('tweet', 'prediction')
    op.drop_column('post', 'prediction')
    # ### end Alembic commands ###
