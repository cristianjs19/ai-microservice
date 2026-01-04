"""Add users and search history tables.

Revision ID: abc123def456
Revises: 68d8eff9c41c
Create Date: 2026-01-04 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "abc123def456"
down_revision: Union[str, None] = "68d8eff9c41c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add users and search_history tables for authentication."""
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("phone", sa.String(length=20), nullable=True),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column(
            "is_superuser", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("phone"),
    )

    # Create indexes for users
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=False)
    op.create_index(op.f("ix_users_phone"), "users", ["phone"], unique=False)

    # Create search_history table
    op.create_table(
        "search_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("transformed_query", sa.Text(), nullable=True),
        sa.Column("channel_id", sa.String(length=100), nullable=True),
        sa.Column("top_k", sa.Integer(), nullable=False, server_default=sa.text("5")),
        sa.Column(
            "similarity_threshold",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0.7"),
        ),
        sa.Column(
            "results_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "processing_time_ms",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_search_history_user_id_users"),
            ondelete="CASCADE",
        ),
    )

    # Create indexes for search_history
    op.create_index(
        op.f("ix_search_history_user_id"),
        "search_history",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_search_history_created_at"),
        "search_history",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove users and search_history tables."""
    # Drop search_history table and indexes
    op.drop_index(op.f("ix_search_history_created_at"), table_name="search_history")
    op.drop_index(op.f("ix_search_history_user_id"), table_name="search_history")
    op.drop_table("search_history")

    # Drop users table and indexes
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_index(op.f("ix_users_phone"), table_name="users")
    op.drop_table("users")
