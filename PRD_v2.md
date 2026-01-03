# PRD for AI Processing Service (Version 2.1 - Revised)

## 1. Project Overview

### 1.1 Purpose

Develop a FastAPI microservice that processes raw video transcripts, formats them into human-readable articles, creates vector embeddings, and provides a RAG-based search API. This service will ingest data triggered by RabbitMQ events from a Fetching Service and serve search queries.

### 1.2 Scope

**Core Functionality:**
- Ingest raw transcript data via RabbitMQ
- Format transcripts using an LLM (AI Agent 1)
- Chunk formatted text and generate embeddings using OpenRouter's embedding API
- Store formatted text and embeddings in PostgreSQL/PGVector
- Provide a REST API for RAG queries with context-aware results
- Implement query validation/transformation using an LLM (AI Agent 2)
- AI Orchestration: LangChain/LangGraph
- Data Storage: PostgreSQL (PGVector)
- Messaging: RabbitMQ (aio-pika)
- Deployment: Dockerized, designed for docker-compose
- Target Scale: Capable of processing hundreds of videos per day

---

## 2. Technical Stack

### 2.1 Core Technologies

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| Framework | FastAPI | >=0.115.0 | Asynchronous web framework |
| Python | Python | 3.12 | |
| Database | PostgreSQL w/ PGVector | 15+ | Stores metadata and vector embeddings |
| ORM | SQLAlchemy (async) | >=2.0.0 | For database interaction |
| Migrations | Alembic | latest | Database schema management |
| Messaging | RabbitMQ | 3-management | For inter-service communication (EDA) |
| AI Orchestration | LangChain / LangGraph | >=0.3.0 | For building AI agents and RAG chains |
| AI Model Access | langchain-openai | >=0.2.0 | Unified interface for OpenAI compatible APIs (incl. OpenRouter) |
| HTTP Client | httpx | >=0.28.0 | For calling Fetching Service REST API |
| Package Installer | uv | latest | Faster dependency installation during build |

### 2.2 AI Models & Providers

| Task | Provider/Model | Dimensions | Notes |
|------|---|---|---|
| Embeddings | OpenRouter API: baai/bge-base-en-v1.5 | 768 | Max 512 tokens input, high-quality, efficient, API-based |
| LLM (Formatting) | OpenRouter API: openai/gpt-oss-120b | N/A | Used for text formatting (AI Agent 1) |
| LLM (Query Guardrail) | OpenRouter API: openai/gpt-oss-120b | N/A | Used for validating/rewriting user queries (AI Agent 2) |

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

# Testing
pytest>=8.0.0
pytest-asyncio>=0.24.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
respx>=0.21.0
polyfactory>=2.15.0
testcontainers>=4.0.0
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
│   │   ├── fetching_client.py  # HTTP client for Fetching Service API
│   │   ├── formatting_service.py # LLM for text cleaning
│   │   ├── chunking_service.py # Text splitting logic
│   │   ├── embedding_service.py  # Interface to embedding model
│   │   └── rag_service.py      # RAG search, query transformation
│   └── api/                    # FastAPI Endpoints
│       ├── __init__.py
│       ├── deps.py             # Dependency injection (DB session, services)
│       ├── v1/
│       │   ├── __init__.py
│       │   ├── schemas/
│       │   │   ├── __init__.py
│       │   │   └── search.py   # Request/Response schemas
│       │   └── endpoints/
│       │       ├── __init__.py
│       │       ├── search.py   # /api/v1/search endpoint
│       │       └── stats.py    # /api/v1/stats endpoint
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
│   │   ├── test_chunking_service.py
│   │   └── test_rag_service.py
│   ├── integration/            # Integration tests (API endpoints, DB, RabbitMQ)
│   │   ├── __init__.py
│   │   └── test_search_api.py
│   └── e2e/                    # End-to-end tests (full pipeline)
│       └── test_full_pipeline.py
├── .env.docker                 # Environment variables for Docker Compose
├── Dockerfile                  # Main Dockerfile for the service
├── docker-compose.yml          # Local development setup
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

    # Stores: video_title, channel_name, published_at, duration_seconds
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

## 5. Inter-Service Communication

### 5.0 Communication Architecture

**Fetching Transcript Data:**
When the AI Service receives a `transcript.fetched` message, it retrieves the raw transcript and metadata via REST API:

- **Endpoint:** `GET {FETCHING_SERVICE_URL}/videos/{video_id}/caption`
- **Response:** Plain text (raw transcript)
- **Additional Metadata Endpoint:** `GET {FETCHING_SERVICE_URL}/videos/{video_id}`
- **Response Structure (from Fetching Service):**
```json
{
  "video_id": "abc123",
  "title": "How to succeed in life",
  "url": "https://youtube.com/watch?v=abc123",
  "channel_id": "UC...",
  "channel": {
    "name": "Example Channel",
    "url": "https://youtube.com/channel/UC..."
  },
  "upload_date": "2024-01-15T10:30:00Z",
  "view_count": 150000,
  "duration": 1800,
  "transcription": "Full transcript text...",
  "transcription_language": "en",
  "has_transcription": true
}
```

**Error Handling:**
- If API call fails (network error, 404, etc.), update VideoDocument status to FAILED
- Implement exponential backoff retry (4 attempts)
- Log error details to error_message field

**Environment Variable:**
- `FETCHING_SERVICE_URL=http://fetching-service:8000`

**Note:** The AI Service does NOT publish events back to RabbitMQ since the Fetching Service doesn't need to know about processing completion status.

---

## 6. Core Services & Logic

### 6.1 Fetching Service Client

**File:** app/services/fetching_client.py

**Purpose:** HTTP client wrapper for calling Fetching Service REST API

**Methods:**
- `async get_video_metadata(video_id: str) -> dict`: Fetch full video details
- `async get_video_caption(video_id: str) -> str`: Fetch only transcript text

**Implementation:**
```python
import httpx
from app.config import settings

class FetchingServiceClient:
    def __init__(self):
        self.base_url = settings.FETCHING_SERVICE_URL
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_video_metadata(self, video_id: str) -> dict:
        """Fetch complete video metadata including channel info."""
        response = await self.client.get(f"{self.base_url}/videos/{video_id}")
        response.raise_for_status()
        return response.json()
    
    async def get_video_caption(self, video_id: str) -> str:
        """Fetch raw transcript text only."""
        response = await self.client.get(f"{self.base_url}/videos/{video_id}/caption")
        response.raise_for_status()
        return response.text
    
    async def close(self):
        await self.client.aclose()
```

### 6.2 Ingestion Pipeline (RabbitMQ Consumer)

**File:** app/consumers/video_processor.py

**Trigger:** Listens to `transcript.fetched` queue on RabbitMQ.

**Message Schema:**
```json
{
  "video_id": "abc123"
}
```
*Note: Just video_id is sufficient since we can query the Fetching Service API for all other data*

**Process Flow:**
1. **Receive message** from `transcript.fetched` queue
2. **Check duplicate**: Query `video_documents` table by `source_video_id`. If exists, skip processing
3. **Fetch video data**: 
   - Call `FetchingServiceClient.get_video_metadata(video_id)` to get full details
   - Call `FetchingServiceClient.get_video_caption(video_id)` to get raw transcript
4. **Create VideoDocument record** (status: `PENDING`)
   - Store `source_video_id` = video_id
   - Store `source_channel_id` = channel_id from metadata
   - Store in `meta_data`: `{"video_title": "...", "channel_name": "...", "published_at": "...", "duration_seconds": 1800}`
5. **Call Formatting Service**: Pass raw transcript text → Get formatted content
6. **Update VideoDocument**: 
   - Set `formatted_content` = formatted text
   - Set status: `PROCESSING`
7. **Call Chunking Service**: Pass `formatted_content` → Get list of chunk strings
8. **Call Embedding Service**: Pass list of chunks → Get list of embedding vectors
9. **Persist VideoChunks**: Create VideoChunk records with content, chunk_index, embedding, document_id
10. **Update VideoDocument**: Set status to `COMPLETED`

**Error Handling:**
- Wrap each step in try-except blocks
- On any error:
  - Update VideoDocument status to `FAILED`
  - Store exception message in `error_message` field
  - Send message to dead-letter queue for manual review
- Implement **exponential backoff retry policy**:
  - Max retries: 4
  - Base delay: 2 seconds
  - Backoff factor: 2 (delays: 2s, 4s, 8s, 16s)
  - Only retry on transient errors (network issues, timeouts)
  - Don't retry on 404 or validation errors

### 6.3 Formatting Service (AI Agent 1)

**File:** app/services/formatting_service.py

**Input:** Raw transcript string

**Logic:**
1. Instantiate `ChatOpenAI` pointing to OpenRouter for `openai/gpt-oss-120b`
2. Use LangChain `ChatPromptTemplate` with the following system prompt:

```
You are an expert transcript editor. Your task is to format raw video captions into a highly readable, engaging article format. 

NO SUMMARIZATION: Retain all original information.
PARAGRAPH BREAKS: Insert logical breaks (\n\n) where topics shift.
PUNCTUATION: Fix run-ons, add commas/periods.
CLEAN UP: Remove verbal fillers ("uh", "um") if they disrupt readability.
FORMATTING: Keep as prose, no headers/bullets.
```

3. Execute LLM call with error handling and retry logic
4. **Output:** Formatted text string

**Error Handling:**
- Timeout after 60 seconds
- Retry on rate limits (exponential backoff, max 3 retries)
- Log API errors for debugging

### 6.4 Chunking Service

**File:** app/services/chunking_service.py

**Input:** Formatted text string

**Logic:**
1. Use `RecursiveCharacterTextSplitter` from LangChain
2. Use `tiktoken` for token-based chunking (bge-base-en-v1.5 has 512 token limit)
3. Configuration:
   - **chunk_size**: 400 tokens (safe margin below 512 token limit)
   - **chunk_overlap**: 50 tokens (preserves context between chunks)
   - **separators**: `["\n\n", "\n", ". ", " "]` (respects LLM paragraphs first)
   - **length_function**: Use tiktoken's `cl100k_base` encoding for accurate token counting
4. Process:
   - Split formatted_content into chunks
   - Assign sequential `chunk_index` (0, 1, 2, ...)
   - Return list of tuples: `[(chunk_index, content), ...]`

**Why Token-Based?**
- Character-based chunking (1000 chars) could exceed 512 tokens
- Token-based ensures compatibility with embedding model
- 400 tokens provides safety buffer for variation in token density

**Note on Token Counting:** We use tiktoken's `cl100k_base` encoding as an approximation. While BGE models use their own tokenizer, this approximation is acceptable for chunking purposes and avoids adding HuggingFace tokenizers as a dependency.

**Example:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

def chunk_text(formatted_text: str) -> List[Tuple[int, str]]:
    encoding = tiktoken.get_encoding("cl100k_base")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=lambda text: len(encoding.encode(text)),
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = splitter.split_text(formatted_text)
    return [(i, chunk) for i, chunk in enumerate(chunks)]
```

**Output:** List of `(chunk_index, content)` tuples

### 6.5 Embedding Service

**File:** app/services/embedding_service.py

**Input:** List of chunk content strings

**Logic:**
1. Instantiate `OpenAIEmbeddings` from `langchain-openai`
   - `model="baai/bge-base-en-v1.5"`
   - `openai_api_key=os.getenv("OPENROUTER_API_KEY")`
   - `openai_api_base="https://openrouter.ai/api/v1"`
2. Call `embeddings.embed_documents(list_of_chunk_content)`
3. Validate: Each embedding should be 768-dimensional vector
4. **Output:** List of embedding vectors `[[0.1, 0.2,...], [0.3, 0.4,...]]`

**Error Handling:**
- Validate vector dimensions (must be 768)
- Handle API rate limits with exponential backoff
- Batch large requests if needed (though OpenRouter handles batching)

### 6.6 RAG Service (Querying)

**File:** app/services/rag_service.py

**Endpoint:** `POST /api/v1/search`

**Input:** User query (query: str), optional channel_id: str, top_k: int=5, similarity_threshold: float=0.7

**Process Flow:**

#### Step 1: Query Guardrail (AI Agent 2)
- **Input:** User query string
- **LLM Call:** Use `openai/gpt-oss-120b` via OpenRouter
- **Prompt:**
```
Analyze the following search query and determine if it's valid for semantic search.

Rules:
1. GARBAGE: Random characters, gibberish (e.g., "Mytj", "asdfgh") → Return INVALID
2. INCOMPLETE: Sentence fragments without clear intent (e.g., "I want to know about") → Return INCOMPLETE
3. VAGUE: Single words or unclear phrases (e.g., "death", "success") → Rewrite to semantic sentence
4. GOOD: Clear, complete questions or statements → Pass through unchanged

Query: {user_query}

Respond ONLY with JSON:
{
  "status": "OK" | "INVALID" | "INCOMPLETE",
  "transformed_query": "<semantic query>" (required if status=OK),
  "error_message": "<helpful message>" (required if status!=OK)
}
```

- **Response Processing:**
  - If `status == "INVALID"`: Return 400 error to user with message
  - If `status == "INCOMPLETE"`: Return 400 error asking for more detail
  - If `status == "OK"`: Use `transformed_query` for vector search

**Example Transformations:**
- Input: "death" → Output: "What are the speaker's philosophical views on death and mortality?"
- Input: "success" → Output: "What advice or strategies does the speaker provide for achieving success?"
- Input: "How to succeed in life" → Output: "How to succeed in life" (no change needed)

#### Step 2: Embed Query
- Use the same `OpenAIEmbeddings` instance as in 6.5
- Embed the `transformed_query` string
- Get 768-dimensional query vector

#### Step 3: Vector Search
- Query PostgreSQL PGVector using cosine similarity
- SQL approach:
```sql
SELECT 
    vc.id,
    vc.content,
    vc.chunk_index,
    vd.source_video_id,
    vd.source_channel_id,
    vd.meta_data,
    (vc.embedding <=> :query_vector) as similarity_score
FROM video_chunks vc
JOIN video_documents vd ON vc.document_id = vd.id
WHERE vd.status = 'COMPLETED'
  AND (:channel_id IS NULL OR vd.source_channel_id = :channel_id)
  AND (vc.embedding <=> :query_vector) < (1 - :similarity_threshold)
ORDER BY similarity_score ASC
LIMIT :top_k_chunks
```

**Note on Similarity Threshold:**
- PGVector uses distance operators: `<=>` (cosine distance = 1 - cosine similarity)
- BGE models produce similarity scores in [0.6, 1.0] range
- Default threshold: 0.7 (meaning distance < 0.3)
- This filters out low-quality matches

#### Step 4: Fetch Context (Neighbor Segments)
For each matching chunk, fetch neighboring chunks for context:
- Fetch `chunk_index - 1` (previous segment)
- Current chunk (already have)
- Fetch `chunk_index + 1` (next segment)

**Query:**
```sql
SELECT content, chunk_index
FROM video_chunks
WHERE document_id = :document_id
  AND chunk_index IN (:current_index - 1, :current_index, :current_index + 1)
ORDER BY chunk_index ASC
```

#### Step 5: Result Construction
Group results by `source_video_id` (since multiple chunks may match from same video):

**Output:** JSON response with relevant video candidates and text segments

---

## 7. API Endpoints

### 7.1 Core Endpoints (AI Service)

| Method | Endpoint | Description | Request Body Schema | Response Schema | Triggered By |
|--------|----------|-------------|-------------------|-----------------|--------------|
| POST | /api/v1/search | Perform RAG query, return relevant video candidates and text segments | SearchQueryRequest | SearchResponse | User Query |
| POST | /api/v1/process-video-by-id | Manually trigger processing for a given source_video_id (for testing/debugging) | ProcessVideoRequest | ProcessStatusResponse | Manual Trigger / Debugging |
| GET | /api/v1/stats | Service statistics (processed count, pending count, failed count) | N/A | StatsResponse | Admin/Monitoring |

### 7.2 Internal/System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check for the service |
| GET | /ready | Readiness check (DB connection, RabbitMQ connection) |

---

## 8. Data Models (Schemas)

### 8.1 Request Schemas

**File:** app/api/v1/schemas/search.py

```python
from typing import Optional
from pydantic import BaseModel, Field

class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="The user's search query.")
    channel_id: Optional[str] = Field(None, description="Optional. Filter results to a specific YouTube channel ID.")
    top_k: int = Field(5, gt=0, le=20, description="Maximum number of video candidates to return.")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score (0.7 recommended for BGE models).")

class ProcessVideoRequest(BaseModel):
    source_video_id: str = Field(..., description="The YouTube video ID to process.")
```

### 8.2 Response Schemas

**File:** app/api/v1/schemas/search.py

```python
from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class QueryTransformationResult(BaseModel):
    """Result from Query Guardrail (AI Agent 2)."""
    status: Literal["OK", "INVALID", "INCOMPLETE"]
    transformed_query: Optional[str] = None
    error_message: Optional[str] = None

class ChunkContext(BaseModel):
    """Context segments surrounding the matched chunk."""
    previous_segment: Optional[str] = Field(None, description="Content from chunk_index - 1")
    current_segment: str = Field(..., description="The matched chunk content")
    next_segment: Optional[str] = Field(None, description="Content from chunk_index + 1")

class SearchResultItem(BaseModel):
    video_id: str
    similarity_score: float = Field(..., description="Cosine similarity score (0-1, higher is better)")
    
    # Context segments
    context: ChunkContext
    
    # Metadata from VideoDocument.meta_data JSONB field
    video_title: Optional[str] = None
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    published_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None

class SearchResponse(BaseModel):
    query: str = Field(..., description="Original user query")
    transformed_query: str = Field(..., description="Query after guardrail transformation")
    results: List[SearchResultItem] = Field(..., description="Ranked list of relevant video segments")
    total_results: int = Field(..., description="Number of results returned")

class ProcessStatusResponse(BaseModel):
    source_video_id: str
    status: str  # ProcessingStatus enum as string
    message: Optional[str] = None

class StatsResponse(BaseModel):
    total_documents: int
    completed_documents: int
    pending_documents: int
    processing_documents: int
    failed_documents: int
    total_chunks: int
```

---

## 9. Key Processes & Workflows

### 9.1 Video Processing Workflow (RabbitMQ Triggered)

**Sequence Diagram:**
```
RabbitMQ → AI Service → Fetching Service API → AI Service DB
```

**Detailed Steps:**

1. **Message Received:** `transcript.fetched` event with `{ "video_id": "abc123" }`
2. **Duplicate Check:** Query `video_documents` WHERE `source_video_id = 'abc123'`. If exists, skip
3. **Fetch Data:**
   - HTTP GET `/videos/abc123` → Get metadata
   - HTTP GET `/videos/abc123/caption` → Get raw transcript
4. **Create Record:** INSERT VideoDocument (status: `PENDING`)
5. **Format Text:** LLM Agent 1 → formatted_content
6. **Update Status:** UPDATE VideoDocument (status: `PROCESSING`)
7. **Chunk Text:** Split formatted_content into 400-token chunks
8. **Generate Embeddings:** API call to OpenRouter → 768-dim vectors
9. **Persist Chunks:** INSERT VideoChunks (content, chunk_index, embedding)
10. **Finalize:** UPDATE VideoDocument (status: `COMPLETED`)

**Error Flow:**
- On any exception: UPDATE VideoDocument (status: `FAILED`, error_message)
- Publish to dead-letter queue
- Log error details

### 9.2 RAG Query Workflow (API Triggered)

**Sequence Diagram:**
```
User → AI Service API → Query Guardrail (LLM Agent 2) → Vector DB → Context Fetch → Response
```

**Detailed Steps:**

1. **Receive Request:** POST /api/v1/search with query, filters
2. **Query Guardrail:** 
   - LLM Agent 2 analyzes query
   - If INVALID/INCOMPLETE → Return 400 error
   - If OK → Get transformed_query
3. **Embed Query:** OpenRouter API → 768-dim query vector
4. **Vector Search:** 
   - Query PGVector with cosine similarity
   - Filter by channel_id if provided
   - Apply similarity_threshold (distance < 1 - threshold)
   - Get top_k matching chunks
5. **Fetch Context:** For each match, get neighboring chunks (index ±1)
6. **Group Results:** Combine chunks from same video
7. **Enrich Metadata:** Extract from VideoDocument.meta_data
8. **Return Response:** SearchResponse JSON

---

## 10. Error Handling & Resilience

### 10.1 RabbitMQ Consumer Error Handling

**Retry Strategy:**
- **Max Retries:** 4
- **Backoff Policy:** Exponential (2s, 4s, 8s, 16s)
- **Retry Conditions:** Network errors, timeouts, transient failures
- **No Retry Conditions:** 404 errors, validation errors, malformed messages

**Dead Letter Queue (DLQ):**
- Failed messages after 4 retries go to `transcript.failed` queue
- Manual inspection and reprocessing capability
- Store failure metadata (attempt count, error messages, timestamps)

### 10.2 LLM/Embedding API Error Handling

**Timeout Configuration:**
- Formatting LLM: 60 seconds
- Query Guardrail LLM: 30 seconds
- Embedding API: 45 seconds

**Retry Logic:**
- Rate Limit Errors (429): Exponential backoff, max 3 retries
- Server Errors (5xx): Exponential backoff, max 2 retries
- Authentication Errors (401): No retry, log and fail immediately
- Invalid Input (400): No retry, log validation error

**Circuit Breaker Pattern:**
- After 5 consecutive failures, pause processing for 5 minutes
- Prevents cascading failures during API outages
- Resume gradually (half-open state)

### 10.3 Database Error Handling

**Connection Pool:**
- Min connections: 5
- Max connections: 20
- Connection timeout: 10 seconds
- Connection max age: 1 hour

**Transaction Management:**
- Use SQLAlchemy's async session with automatic rollback on errors
- For multi-step operations (create document + chunks), use explicit transactions
- Implement savepoints for partial rollback capability

**Constraint Violations:**
- Duplicate `source_video_id`: Skip processing (idempotency)
- Foreign key violations: Log error and fail gracefully
- Null constraint violations: Validate data before insert

### 10.4 API Endpoint Error Handling

**HTTP Status Codes:**
- 200: Success
- 400: Bad request (invalid query, failed guardrail)
- 404: Resource not found
- 422: Validation error (Pydantic)
- 429: Rate limit exceeded
- 500: Internal server error
- 503: Service unavailable (DB/RabbitMQ down)

**Error Response Format:**
```json
{
  "error": {
    "code": "INVALID_QUERY",
    "message": "Query failed guardrail validation",
    "details": "Query appears to be incomplete. Please provide more context."
  }
}
```

**Logging:**
- Use structured logging (JSON format)
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Include correlation IDs for request tracing
- Sensitive data (API keys) never logged

---

## 11. Configuration Management

### 11.1 Environment Variables (.env.docker)

```bash
# Database
DATABASE_URL=postgresql+asyncpg://ai_user:password@postgres:5432/ai_service
POSTGRES_USER=ai_user
POSTGRES_PASSWORD=password
POSTGRES_DB=ai_service

# RabbitMQ
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/
RABBITMQ_QUEUE_NAME=transcript.fetched
RABBITMQ_DLQ_NAME=transcript.failed
RABBITMQ_EXCHANGE=video.events

# Fetching Service
FETCHING_SERVICE_URL=http://fetching-service:8000

# OpenRouter API
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# AI Models
EMBEDDING_MODEL_NAME=baai/bge-base-en-v1.5
EMBEDDING_DIMENSIONS=768
LLM_MODEL_NAME=openai/gpt-oss-120b
QUERY_GUARDRAIL_MODEL=openai/gpt-oss-120b

# RAG Configuration
TOP_K_DEFAULT=5
SIMILARITY_THRESHOLD_DEFAULT=0.7
MAX_CHUNK_SIZE_TOKENS=400
CHUNK_OVERLAP_TOKENS=50

# API Configuration
API_V1_PREFIX=/api/v1
API_TITLE=AI Processing Service
API_VERSION=2.1.0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Service Configuration
SERVICE_NAME=ai-processing-service
ENVIRONMENT=production
```

### 11.2 Settings Class (app/config.py)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env.docker",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Database
    database_url: str
    
    # RabbitMQ
    rabbitmq_url: str
    rabbitmq_queue_name: str = "transcript.fetched"
    rabbitmq_dlq_name: str = "transcript.failed"
    rabbitmq_exchange: str = "video.events"
    
    # Fetching Service
    fetching_service_url: str
    
    # OpenRouter
    openrouter_api_key: str
    openrouter_api_base: str = "https://openrouter.ai/api/v1"
    
    # Models
    embedding_model_name: str = "baai/bge-base-en-v1.5"
    embedding_dimensions: int = 768
    llm_model_name: str = "openai/gpt-oss-120b"
    query_guardrail_model: str = "openai/gpt-oss-120b"
    
    # RAG
    top_k_default: int = 5
    similarity_threshold_default: float = 0.7
    max_chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50
    
    # API
    api_v1_prefix: str = "/api/v1"
    api_title: str = "AI Processing Service"
    api_version: str = "2.1.0"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Service
    service_name: str = "ai-processing-service"
    environment: str = "production"

# Singleton instance
settings = Settings()
```

### 11.3 Docker Compose Configuration

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_USER: ai_user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: ai_service
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
  
  ai-service:
    build: .
    env_file: .env.docker
    depends_on:
      - postgres
      - rabbitmq
    ports:
      - "8001:8000"
    volumes:
      - ./app:/app/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

volumes:
  postgres_data:
```

---

> **📚 Additional Reference Material**  
> For testing strategy, deployment operations, performance considerations, security, future enhancements, and glossary, see: [PRD_REFERENCE_APPENDIX.md](PRD_REFERENCE_APPENDIX.md)

---

**END OF PRD v2.1 (Core Implementation Guide)**