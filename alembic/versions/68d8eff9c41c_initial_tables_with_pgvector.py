"""Initial tables with PGVector support.

Revision ID: 68d8eff9c41c
Revises:
Create Date: 2026-01-03 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "68d8eff9c41c"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create processing_status_enum type
    op.execute(
        "CREATE TYPE processing_status_enum AS ENUM ('pending', 'processing', 'completed', 'failed')"
    )

    # Create video_documents table using raw SQL
    op.execute("""
        CREATE TABLE video_documents (
            id UUID PRIMARY KEY,
            source_video_id VARCHAR(255) NOT NULL UNIQUE,
            source_channel_id VARCHAR(255),
            formatted_content TEXT NOT NULL,
            status processing_status_enum NOT NULL DEFAULT 'pending',
            meta_data JSONB NOT NULL DEFAULT '{}',
            error_message TEXT,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE NOT NULL
        )
    """)

    op.create_index(
        "ix_video_documents_source_channel_id",
        "video_documents",
        ["source_channel_id"],
        unique=False,
    )
    op.create_index(
        "ix_video_documents_source_video_id",
        "video_documents",
        ["source_video_id"],
        unique=False,
    )

    # Create video_chunks table with pgvector
    op.execute("""
        CREATE TABLE video_chunks (
            id UUID PRIMARY KEY,
            document_id UUID NOT NULL REFERENCES video_documents(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            embedding vector(768) NOT NULL
        )
    """)

    # Create HNSW index for vector search
    op.execute(
        "CREATE INDEX idx_video_chunks_embedding ON video_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m=16, ef_construction=64)"
    )


def downgrade() -> None:
    # Drop HNSW index
    op.drop_index("idx_video_chunks_embedding", table_name="video_chunks")

    # Drop video_chunks table
    op.drop_table("video_chunks")

    # Drop video_documents table
    op.drop_index("ix_video_documents_source_video_id", table_name="video_documents")
    op.drop_index("ix_video_documents_source_channel_id", table_name="video_documents")
    op.drop_table("video_documents")

    # Drop enum type
    op.execute("DROP TYPE processing_status_enum")
