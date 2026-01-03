# PRD Reference Appendix

> **Note:** This document contains supplementary reference material extracted from [PRD_v2.md](PRD_v2.md).  
> These sections are useful for context but not required during active implementation phases.

---

## 12. Testing Strategy

### 12.1 Test Categories

**Unit Tests (tests/unit/):**
- Individual service functions (formatting, chunking, embedding logic)
- Query transformation logic
- Mock all external dependencies (LLM APIs, DB, HTTP clients)
- Fast execution (< 1 second per test)

**Integration Tests (tests/integration/):**
- API endpoints with real DB (testcontainers)
- RabbitMQ message handling
- Service-to-service communication (mock Fetching Service API)
- Database operations with real PostgreSQL + PGVector

**E2E Tests (tests/e2e/):**
- Full pipeline simulation:
  1. Publish RabbitMQ message
  2. Verify DB state (VideoDocument created)
  3. Call search API
  4. Verify results structure and relevance
- Use testcontainers for all infrastructure
- Mock external APIs (OpenRouter)

### 12.2 Test Infrastructure

**Fixtures (tests/conftest.py):**
```python
import pytest
from testcontainers.postgres import PostgresContainer
from testcontainers.rabbitmq import RabbitMqContainer

@pytest.fixture(scope="session")
async def postgres_container():
    with PostgresContainer("pgvector/pgvector:pg15") as postgres:
        yield postgres

@pytest.fixture(scope="session")
async def rabbitmq_container():
    with RabbitMqContainer("rabbitmq:3-management") as rabbitmq:
        yield rabbitmq

@pytest.fixture
async def db_session(postgres_container):
    # Create async engine and session
    # Run migrations
    # Yield session
    # Cleanup after test
    pass
```

**Mocking External APIs:**
- Use `respx` to mock OpenRouter API calls
- Use `pytest-mock` for service method mocking

**Test Data Factories (tests/factories/):**
```python
from polyfactory.factories import SQLAlchemyFactory
from app.models.videos import VideoDocument, VideoChunk

class VideoDocumentFactory(SQLAlchemyFactory[VideoDocument]):
    __model__ = VideoDocument

class VideoChunkFactory(SQLAlchemyFactory[VideoChunk]):
    __model__ = VideoChunk
```

### 12.3 Test Coverage Goals

- Unit Tests: > 90% coverage
- Integration Tests: All API endpoints, RabbitMQ consumers
- E2E Tests: At least 2 critical workflows (ingestion + search)

### 12.4 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run specific test file
pytest tests/unit/test_formatting_service.py -v
```

---

## 13. Deployment & Operations

### 13.1 Docker Build

**Dockerfile:**
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv for faster dependency installation
RUN pip install uv

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Expose port
EXPOSE 8000

# Run migrations and start service
CMD alembic upgrade head && \
    uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 13.2 Health Checks

**Endpoints:**
- `/health`: Always returns 200 (service is running)
- `/ready`: Returns 200 if DB and RabbitMQ connections are healthy

**Docker Compose Health Check:**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/ready"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### 13.3 Monitoring & Observability

**Metrics to Track:**
- Processing throughput (videos/hour)
- Average processing time per video
- Error rate by stage (formatting, chunking, embedding)
- Search query latency (p50, p95, p99)
- Database connection pool utilization
- RabbitMQ queue depth

**Logging:**
- Structured JSON logs
- Correlation IDs for request tracing
- Log aggregation (e.g., ELK stack, CloudWatch)

**Alerts:**
- High error rate (> 5% in 5 minutes)
- Processing queue backup (> 100 pending videos)
- Database connection errors
- API rate limit hits

---

## 14. Performance Considerations

### 14.1 Throughput Targets

- **Ingestion:** Process 5-10 videos/minute (depends on video length)
- **Search:** < 500ms p95 latency
- **Concurrent Users:** Support 50+ simultaneous search requests

### 14.2 Optimization Strategies

**Batch Processing:**
- Process multiple chunks in single embedding API call (if OpenRouter supports)
- Batch DB inserts for VideoChunks (100 at a time)

**Caching:**
- Cache query embeddings for repeated searches (Redis)
- Cache query transformation results for common queries

**Database Optimization:**
- HNSW index on embedding column (already in schema)
- Consider index on `source_channel_id` for filtered searches
- Periodic VACUUM ANALYZE on large tables

**Connection Pooling:**
- Reuse httpx client for Fetching Service API calls
- PostgreSQL connection pool (5-20 connections)
- RabbitMQ channel pooling

### 14.3 Scalability

**Horizontal Scaling:**
- Multiple AI service instances can consume from same RabbitMQ queue
- Each instance maintains own DB connection pool
- Consider distributed locking for duplicate prevention

**Vertical Scaling:**
- Increase DB connection pool size
- Add read replicas for search queries
- Increase RabbitMQ consumer prefetch count

---

## 15. Security Considerations

### 15.1 API Security

- No authentication required for search API (as per requirements)
- Rate limiting on search endpoint (e.g., 60 requests/minute per IP)
- Input validation using Pydantic
- SQL injection prevention (SQLAlchemy ORM)

### 15.2 Secrets Management

- Store API keys in environment variables
- Never log API keys or sensitive data
- Use Docker secrets in production
- Rotate API keys regularly

### 15.3 Data Privacy

- Transcript data is already public (YouTube)
- No PII stored in database
- GDPR compliance: Provide video deletion endpoint if needed

---

## 16. Future Enhancements

### 16.1 Phase 2 Features

- **Multi-language Support:** Detect transcript language, use multilingual embeddings
- **Semantic Deduplication:** Identify and merge duplicate videos
- **Advanced Ranking:** ML-based reranking of search results
- **User Feedback Loop:** Allow users to rate search result relevance
- **Analytics Dashboard:** Track popular queries, trending topics

### 16.2 Phase 3 Features

- **Real-time Processing:** WebSocket support for live processing status
- **Hybrid Search:** Combine vector search with keyword search (BM25)
- **Query Expansion:** Use LLM to suggest related queries
- **Video Clustering:** Group videos by topic/theme
- **Export API:** Allow users to export search results

---

## 17. References & Documentation

### 17.1 Official Documentation

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL + PGVector](https://github.com/pgvector/pgvector)
- [OpenRouter API Docs](https://openrouter.ai/docs)
- [RabbitMQ aio-pika](https://aio-pika.readthedocs.io/)
- [SQLAlchemy 2.0 Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-base-en-v1.5)

### 17.2 Internal Documentation

- Fetching Service API: See provided documentation
- RabbitMQ Message Schemas: Document in wiki
- Deployment Runbooks: Create for production rollout

### 17.3 Development Resources

- Code Style: PEP 8 (enforced with ruff)
- Type Hints: Required for all functions
- Docstrings: Google style
- Commit Messages: Conventional Commits

---

## 18. Glossary

| Term | Definition |
|------|------------|
| **RAG** | Retrieval-Augmented Generation: Combining vector search with LLM generation |
| **Vector Embedding** | Numerical representation of text in high-dimensional space (768 dims) |
| **Chunking** | Splitting long text into smaller segments for embedding |
| **Semantic Search** | Search based on meaning rather than exact keywords |
| **Query Guardrail** | LLM-based validation/transformation of user queries |
| **Dead Letter Queue** | Queue for messages that failed processing after retries |
| **HNSW** | Hierarchical Navigable Small World: Fast approximate nearest neighbor search |
| **Cosine Similarity** | Measure of similarity between vectors (0-1, higher is more similar) |
| **PGVector** | PostgreSQL extension for vector operations |
| **EDA** | Event-Driven Architecture: System design using asynchronous messages |

---

## Appendix A: Example Workflows

### A.1 Complete Ingestion Flow

```
1. Fetching Service fetches transcript → Publishes { "video_id": "abc123" } to RabbitMQ
2. AI Service consumes message
3. Check duplicate: SELECT * FROM video_documents WHERE source_video_id = 'abc123'
   → Not found, continue
4. HTTP GET fetching-service:8000/videos/abc123 → Get metadata
5. HTTP GET fetching-service:8000/videos/abc123/caption → Get transcript
6. INSERT VideoDocument (status=PENDING)
7. Call LLM Agent 1: Format transcript → "The speaker discusses..."
8. UPDATE VideoDocument (formatted_content=..., status=PROCESSING)
9. Chunk text: Split into 400-token chunks → 15 chunks
10. Embed chunks: API call → 15 x 768-dim vectors
11. INSERT VideoChunks (15 records)
12. UPDATE VideoDocument (status=COMPLETED)
```

### A.2 Complete Search Flow

```
1. User sends: POST /api/v1/search { "query": "death", "top_k": 5 }
2. Call Query Guardrail LLM:
   Input: "death"
   Output: { "status": "OK", "transformed_query": "What are the speaker's views on death?" }
3. Embed query: "What are the speaker's views on death?" → 768-dim vector
4. Vector search:
   - Find 5 closest chunks (cosine similarity > 0.7)
   - Results: chunks from 3 different videos
5. For each chunk, fetch context:
   - Chunk A (index=5): Fetch chunks 4, 5, 6
   - Chunk B (index=12): Fetch chunks 11, 12, 13
   - Chunk C (index=3): Fetch chunks 2, 3, 4
6. Group by video, extract metadata
7. Return: SearchResponse with 3 videos, each with context segments
```

---

## Appendix B: Database Schema SQL

```sql
-- Enable PGVector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create enum type
CREATE TYPE processing_status_enum AS ENUM ('pending', 'processing', 'completed', 'failed');

-- Create video_documents table
CREATE TABLE video_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_video_id VARCHAR(255) UNIQUE NOT NULL,
    source_channel_id VARCHAR(255),
    formatted_content TEXT NOT NULL,
    status processing_status_enum NOT NULL DEFAULT 'pending',
    meta_data JSONB DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_video_documents_source_video_id ON video_documents(source_video_id);
CREATE INDEX idx_video_documents_source_channel_id ON video_documents(source_channel_id);
CREATE INDEX idx_video_documents_status ON video_documents(status);

-- Create video_chunks table
CREATE TABLE video_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES video_documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(768) NOT NULL
);

CREATE INDEX idx_video_chunks_document_id ON video_chunks(document_id);
CREATE INDEX idx_video_chunks_embedding ON video_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

---

**END OF PRD REFERENCE APPENDIX**
