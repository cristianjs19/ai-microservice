# PRD for AI Processing Service (Version 2.0)

## 1. Project Overview

### 1.1 Purpose

Develop a FastAPI microservice that processes raw video transcripts, formats them into human-readable articles, creates vector embeddings, and provides a RAG-based search API. This service will ingest data triggered by RabbitMQ events from a Fetching Service and serve search queries.

### 1.2 Scope

**Core Functionality:**
- Ingest raw transcript data via RabbitMQ
- Format transcripts using an LLM (AI Agent 1)
- Chunk formatted text and generate embeddings using a local or API-based embedding model
- Store formatted text and embeddings in PostgreSQL/PGVector
- Provide a REST API for RAG queries
- Implement query validation/transformation using an LLM (AI Agent 2)
- AI Orchestration: LangChain/LangGraph
- Data Storage: PostgreSQL (PostGIS/PGVector)
- Messaging: RabbitMQ (aio-pika)
- Deployment: Dockerized, designed for docker-compose
- Target Scale: Capable of processing hundreds of videos per day

---

## 2. Technical Stack

### 2.1 Core Technologies

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| Framework | FastAPI | >=0.115.0 | Asynchronous web framework |
| Python | Python | 3.11+ | |
| Database | PostgreSQL w/ PGVector | 15+ | Stores metadata and vector embeddings |
| ORM | SQLAlchemy (async) | >=2.0.0 | For database interaction |
| Migrations | Alembic | latest | Database schema management |
| Messaging | RabbitMQ | 3-management | For inter-service communication (EDA) |
| AI Orchestration | LangChain / LangGraph | >=0.3.0 | For building AI agents and RAG chains |
| AI Model Access | langchain-openai | >=0.2.0 | Unified interface for OpenAI compatible APIs (incl. OpenRouter) |
| Package Installer | uv | latest | Faster dependency installation during build |

### 2.2 AI Models & Providers

| Task | Provider/Model | Dimensions | Notes |
|------|---|---|---|
| Embeddings | OpenRouter API: baai/bge-base-en-v1.5 | 768 | High-quality, efficient, API-based (no local GPU needed) |
| LLM (Formatting) | OpenRouter API: anthropic/claude-3-haiku | N/A | Used for text formatting (AI Agent 1) |
| LLM (Query Guardrail) | OpenRouter API: anthropic/claude-3-haiku | N/A | Used for validating/rewriting user queries (AI Agent 2). Cheaper model |

### 2.3 Production Dependencies (requirements.txt)

```txt
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
sqlalchemy[asyncio]>=2.0.0
asyncpg>=0.30.0
alembic>=1.14.0
pydantic>=2.10.0
pydantic-settings>=2.6.0
python-dotenv>=1.0.0
httpx>=0.28.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langgraph>=0.2.0
tiktoken>=0.7.0
langchain-postgres>=0.0.9
psycopg[binary]>=3.1.0
aio-pika>=9.4.0
# Note: No ML libs like torch/sentence-transformers as embeddings are API-based.

pytest>=8.0.0
pytest-asyncio>=0.24.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
respx>=0.21.0
polyfactory>=2.15.0
testcontainers>=4.0.0 # For spinning up RabbitMQ/Postgres in tests
```

---

## 3. Project Structure

```
ai-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app initialization & lifespan events
│   ├── config.py               # Settings management (pydantic-settings)
│   ├── exceptions.py           # Custom exception classes
│   ├── core/
│   │   ├── __init__.py
│   │   └── database.py         # Async engine + session management, PGVector setup
│   ├── models/                 # SQLAlchemy ORM Models
│   │   ├── __init__.py
│   │   ├── base.py             # Base model class
│   │   └── videos.py           # VideoDocument, VideoChunk, ProcessingStatus enum
│   ├── consumers/              # RabbitMQ message handlers
│   │   ├── __init__.py
│   │   └── video_processor.py  # Listens for transcript.fetched
│   ├── services/               # Core business logic
│   │   ├── __init__.py
│   │   ├── formatting_service.py # LLM for text cleaning
│   │   ├── chunking_service.py # Text splitting logic
│   │   ├── embedding_service.py  # Interface to embedding model
│   │   └── rag_service.py      # RAG search, query transformation
│   └── api/                    # FastAPI Endpoints
│       ├── __init__.py
│       ├── deps.py             # Dependency injection (DB session, services)
│       ├── v1/
│       │   ├── __init__.py
│       │   └── endpoints/
│       │       ├── __init__.py
│       │       └── search.py     # /api/v1/search endpoint
│       └── router.py           # Main API router
├── alembic/                    # Alembic configuration and migrations
│   ├── versions/
│   └── env.py
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures (DB, mocks, factories)
│   ├── factories/              # Data factories (Polyfactory)
│   │   ├── __init__.py
│   │   └── video_models.py     # Factories for VideoDocument/Chunk
│   ├── unit/                   # Unit tests for individual services/components
│   │   ├── __init__.py
│   │   ├── test_formatting_service.py
│   │   └── test_rag_service.py
│   ├── integration/            # Integration tests (API endpoints, DB, RabbitMQ)
│   │   ├── __init__.py
│   │   └── test_search_api.py
│   └── e2e/                    # End-to-end tests (full pipeline)
│       └── test_full_pipeline.py
├── .env.docker                 # Environment variables for Docker Compose
├── Dockerfile                  # Main Dockerfile for the service
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 4. SQLAlchemy Models

### File: app/models/base.py

```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass
```

### File: app/models/videos.py

```python
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.models.base import Base

# ---------------------------------------------------------
# Enums
# ---------------------------------------------------------

class ProcessingStatus(str, Enum):
    """Status of the AI processing pipeline for a video."""
    PENDING = "pending"       # Message received, not started
    PROCESSING = "processing" # LLM is formatting / Embeddings being generated
    COMPLETED = "completed"   # Ready for RAG search
    FAILED = "failed"         # Error occurred


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

class VideoDocument(Base):
    """Represents a fully processed video transcript (The 'Parent')."""

    __tablename__ = "video_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    source_video_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True, 
    )
    
    source_channel_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        index=True,
    )

    formatted_content: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[ProcessingStatus] = mapped_column(
        SAEnum(ProcessingStatus, name="processing_status_enum"),
        default=ProcessingStatus.PENDING,
        nullable=False,
    )

    meta_data: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        server_default='{}',
    )

    error_message: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    chunks: Mapped[List["VideoChunk"]] = relationship(
        "VideoChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<VideoDocument(id={self.id}, video_id={self.source_video_id}, status={self.status})>"


class VideoChunk(Base):
    """Represents a semantic vector chunk (The 'Child')."""

    __tablename__ = "video_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("video_documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # BAAI/bge-base-en-v1.5 = 768 dims
    embedding = mapped_column(Vector(768), nullable=False)

    document: Mapped["VideoDocument"] = relationship(
        "VideoDocument",
        back_populates="chunks",
    )

    __table_args__ = (
        Index(
            "idx_video_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
    )

    def __repr__(self) -> str:
        return f"<VideoChunk(id={self.id}, index={self.chunk_index})>"
```

---

## 5. Core Services & Logic

### 5.1 Ingestion Pipeline (RabbitMQ Consumer)

**File:** app/consumers/video_processor.py

**Trigger:** Listens to transcript.fetched queue on RabbitMQ.

**Process Flow:**
1. Receive message `{ "video_id": "..." }`
2. Fetch raw transcript text from Fetching Service DB (read-only access or API call)
3. Check if source_video_id already exists in video_documents. If so, skip
4. Create VideoDocument record (status: PENDING)
5. Call Formatting Service: Pass raw text
6. Update VideoDocument with formatted_content, meta_data, status: PROCESSING
7. Call Chunking Service: Pass formatted_content
8. Call Embedding Service: Pass chunks
9. Persist VideoChunks to DB
10. Update VideoDocument status to COMPLETED

**Error Handling:** If any step fails, update VideoDocument status to FAILED with error_message

### 5.2 Formatting Service (AI Agent 1)

**File:** app/services/formatting_service.py

**Input:** Raw transcript string

**Logic:**
1. Instantiate ChatOpenAI pointing to OpenRouter for anthropic/claude-3-haiku
2. Use LangChain ChatPromptTemplate with the following system prompt:

```
You are an expert transcript editor. Your task is to format raw video captions into a highly readable, engaging article format. 

NO SUMMARIZATION: Retain all original information.
PARAGRAPH BREAKS: Insert logical breaks (\n\n) where topics shift.
PUNCTUATION: Fix run-ons, add commas/periods.
CLEAN UP: Remove verbal fillers ("uh", "um") if they disrupt readability.
FORMATTING: Keep as prose, no headers/bullets.
```

3. Execute LLM call
4. Output: Formatted text string

### 5.3 Chunking Service

**File:** app/services/chunking_service.py

**Input:** Formatted text string

**Logic:**
1. Use RecursiveCharacterTextSplitter from LangChain
2. Configuration:
   - chunk_size=1000
   - chunk_overlap=150
   - separators=["\n\n", "\n", ". ", " "] (Respects LLM paragraphs first)
3. Output: List of VideoChunk content strings, each with its chunk_index

### 5.4 Embedding Service

**File:** app/services/embedding_service.py

**Input:** List of chunk content strings

**Logic:**
1. Instantiate OpenAIEmbeddings from langchain-openai
   - model="baai/bge-base-en-v1.5"
   - openai_api_key=os.getenv("OPENROUTER_API_KEY")
   - openai_api_base="https://openrouter.ai/api/v1"
2. Call embeddings.embed_documents(list_of_chunk_content)
3. Output: List of embedding vectors (e.g., [[0.1, 0.2,...], [0.3, 0.4,...]])

### 5.5 RAG Service (Querying)

**File:** app/services/rag_service.py

**Endpoint:** POST /api/v1/search

**Input:** User query (query: str), optional channel_id: str, top_k: int=5, similarity_threshold: float=0.5

**Process Flow:**

**Query Guardrail (AI Agent 2):**
- Input: User query
- LLM Call: Use anthropic/claude-3-haiku via OpenRouter
- Prompt: Analyze query. If garbage/incomplete, return {"status": "INVALID"}. If vague, rewrite to a more semantic query. If good, return {"status": "OK", "transformed_query": "..."}
- If status == "INVALID", return 400 error to user

**Embed Query:** Use the same OpenAIEmbeddings instance as in 5.4

**Vector Search:**
- Query PGVector using VideoChunk's embedding column and the query vector
- Filter by document.source_channel_id = channel_id if provided
- Apply similarity_threshold
- Select top_k nearest neighbors
- Join with VideoDocument to get source_video_id, formatted_content, meta_data

**Result Construction:**
- For each found chunk, fetch its content and the source_video_id
- Optionally, fetch surrounding chunks for context (e.g., chunk_index-1, chunk_index, chunk_index+1)
- Format into the specified JSON response structure

**Output:** JSON response with relevant video candidates and text segments

---

## 6. API Endpoints

### 6.1 Core Endpoints (AI Service)

| Method | Endpoint | Description | Request Body Schema | Response Schema | Triggered By |
|--------|----------|-------------|-------------------|-----------------|--------------|
| POST | /api/v1/search | Perform RAG query, return relevant video candidates and text segments | SearchQueryRequest | SearchResponse | User Query |
| POST | /api/v1/process-video-by-id | Manually trigger processing for a given source_video_id (for testing/debugging) | ProcessVideoRequest | ProcessStatusResponse | Manual Trigger / Debugging |

### 6.2 Internal/System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check for the service |

---

## 7. Data Models (Schemas)

### 7.1 Request Schemas

**File:** app/api/v1/schemas/search.py

```python
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field

class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The user's search query.")
    channel_id: Optional[str] = Field(None, description="Optional. Filter results to a specific YouTube channel ID.")
    top_k: int = Field(5, gt=0, le=20, description="Maximum number of video candidates to return.")
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity score for a chunk to be considered relevant.")

class ProcessVideoRequest(BaseModel):
    source_video_id: str = Field(..., description="The YouTube video ID to process.")
```

### 7.2 Response Schemas

**File:** app/api/v1/schemas/search.py

```python
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field
from datetime import datetime

class SearchResultItem(BaseModel):
    video_id: str
    similarity_score: float
    relevant_segment: str
    # Optional context: Add previous/next chunk content if implemented
    # previous_segment: Optional[str] = None
    # next_segment: Optional[str] = None
    # Metadata fetched from VideoDocument
    video_title: Optional[str] = None
    channel_id: Optional[str] = None
    upload_date: Optional[datetime] = None

class SearchResponse(BaseModel):
    query: str
    transformed_query: str
    results: List[SearchResultItem]

class ProcessStatusResponse(BaseModel):
    source_video_id: str
    status: str # ProcessingStatus enum as string
    message: Optional[str] = None # For success or error details
```

### 7.3 Processing Status Enum

Already defined in app/models/videos.py

---

## 8. Key Processes & Workflows

### 8.1 Video Processing Workflow (RabbitMQ Triggered)

1. **Message Received:** transcript.fetched event with `{ "video_id": "...", "source_channel_id": "..." }`
2. **State Management:**
   - Create VideoDocument (status: PENDING)
   - Fetch raw text
3. **Formatting:** Call Formatting Service (LLM Agent 1)
4. **Update:** VideoDocument (content, status: PROCESSING)
5. **Chunking:** Call Chunking Service
6. **Embedding:** Call Embedding Service
7. **Persistence:** Save VideoChunks. Update VideoDocument (status: COMPLETED)
8. **Error Handling:** On failure, update VideoDocument (status: FAILED, error_message)

### 8.2 RAG Query Workflow (API Triggered)

1. **Receive Request:** POST /api/v1/search with query, optional filters
2. **Query Guardrail:** Call Query Transformation Service (LLM Agent 2)
   - If query invalid, return 400
3. **Embed Query:** Use OpenAIEmbeddings
4. **Vector Search:** Query video_chunks table (using HNSW index), apply filters and threshold
5. **Retrieve Parent Docs:** Join with video_documents to get metadata
6. **Format Results:** Construct SearchResponse JSON, including relevant segments
7. **Return Response:** Send JSON to user

---

## 9. Error Handling & Resilience

- **RabbitMQ Consumer:** Implement dead-letter queues for failed messages. Retry logic for transient errors (e.g., network issues)
- **LLM/Embedding Calls:** Use try-except blocks for API errors (timeouts, rate limits, invalid keys). Implement exponential backoff for retries
- **Database Operations:** Use SQLAlchemy's async capabilities and try-except for connection errors or constraint violations
- **API Endpoints:** Return appropriate HTTP status codes (4xx for client errors, 5xx for server errors). Log errors comprehensively
- **VideoDocument.status:** Use this field to track and retry failed processing jobs

---

## 10. Configuration Management

### 10.1 Environment Variables (.env.docker files)

- **DATABASE_URL:** Connection string for Postgres
- **RABBITMQ_URL:** Connection string for RabbitMQ
- **OPENROUTER_API_KEY:** API key for OpenRouter
- **OPENAI_API_BASE:** https://openrouter.ai/api/v1
- **EMBEDDING_MODEL_NAME:** baai/bge-base-en-v1.5
- **LLM_MODEL_NAME:** anthropic/claude-3-haiku
- **TOP_K_DEFAULT:** Default value for top_k search parameter
- **SIMILARITY_THRESHOLD_DEFAULT:** Default value for similarity_threshold

### 10.2 Settings Class (app/config.py)

- Load environment variables using Pydantic Settings
- Define constants for models, dimensions, and default API parameters

---

## 11. Testing Strategy

### 11.1 Test Categories

- **Unit Tests:** Individual services (formatting, chunking, embedding, RAG logic). Mocking LLM/DB calls
- **Integration Tests:** API endpoints, service integrations (e.g., RAG search with mocked DB/LLM)
- **E2E Tests:** Full pipeline simulation (publish RabbitMQ message → verify DB state, call search API → verify results)

### 11.2 Test Infrastructure

- Use testcontainers-python to spin up temporary PostgreSQL and RabbitMQ instances
- Use respx or httpx-mock to mock external API calls (OpenRouter)
- Use polyfactory for generating test data

---

## 12. References

- [LangChain Documentation](https://docs.langchain.com/)
- [LangGraph Documentation](https://langgraph.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL + PGVector](https://github.com/pgvector/pgvector)
- [LangChain PGVector](https://python.langchain.com/docs/integrations/vectorstores/pgvector)
- [OpenRouter API Docs](https://openrouter.ai/docs)
- [RabbitMQ aio-pika](https://aio-pika.readthedocs.io/)
- [SQLAlchemy 2.0 Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)