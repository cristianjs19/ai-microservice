# Implementation Checklist: AI Processing Service

This checklist provides atomic tasks for incremental AI agent implementation.  
Reference: [PRD_v2.md](PRD_v2.md) for detailed specifications.

---

## Phase 1: Project Setup & Configuration

**Goal**: Create project scaffolding, dependencies, and configuration.

- [ ] **1.1** Create folder structure as defined in PRD Section 3 (`app/`, `app/core/`, `app/models/`, `app/consumers/`, `app/services/`, `app/api/v1/schemas/`, `app/api/v1/endpoints/`, `tests/`, `alembic/`)
- [x] **1.2** Create `requirements.txt` from PRD Section 2.3 (production deps only)
- [x] **1.3** Create `requirements-dev.txt` with testing dependencies (pytest, respx, polyfactory, testcontainers)
- [x] **1.4** Update `sample.env` with all required variables from PRD Section 11.1
- [ ] **1.5** Create `.gitignore` for Python/FastAPI project (must include `.env`, `__pycache__/`, `.venv/`, `*.pyc`)
- [ ] **1.6** Create `app/config.py` with Settings class from PRD Section 11.2
- [ ] **1.7** Create `app/exceptions.py` with custom exceptions (ProcessingError, FetchingServiceError, EmbeddingError, QueryGuardrailError)
- [ ] **1.8** Create all `__init__.py` files in the folder structure

**Verification**: 
1. Run `pip install -r requirements.txt` successfully
2. Copy `sample.env` to `.env` and add real values
3. Import Settings: `python -c "from app.config import Settings; s=Settings(); print(s.database_url)"`

**Note**: The project infrastructure will be created with docker-compose, so for the moment avoid installing the packages and the verifications.

---

## Phase 2: Database Layer

**Goal**: Set up async SQLAlchemy models with PGVector support. Alembic will inspect these models to generate migrations.

- [ ] **2.1** Relocate existing `base.py` from root to `app/models/base.py` (file exists, just move and update import paths)
- [ ] **2.2** Relocate existing `video_models.py` from root to `app/models/videos.py` (file exists, just move and update import paths)
- [ ] **2.3** Create `app/models/__init__.py` exporting Base, VideoDocument, VideoChunk, ProcessingStatus
- [ ] **2.4** Create `app/core/database.py` with async engine, session factory, and PGVector extension setup (see PRD Section 11.3 docker-compose for connection settings)
- [ ] **2.5** Create `app/core/__init__.py`
- [ ] **2.6** Initialize Alembic: `alembic init alembic`
- [ ] **2.7** Configure `alembic/env.py` for async SQLAlchemy with PGVector (import models, set target_metadata)
- [ ] **2.8** Generate initial migration: `alembic revision --autogenerate -m "Initial tables with PGVector"`
- [ ] **2.9** Run migration: `alembic upgrade head`

**Verification**:
1. Tables `video_documents` and `video_chunks` exist in PostgreSQL
2. PGVector extension enabled: `SELECT * FROM pg_extension WHERE extname = 'vector';`
3. HNSW index on embeddings column exists

**Note**: Connection to DB is probably not working yet, so for the moment avoid the verifications.

---

## Phase 3: HTTP Client (Fetching Service)

**Goal**: Implement HTTP client to communicate with the Fetching Service (yt-scraper).

- [ ] **3.1** Create `app/services/fetching_client.py` with `FetchingServiceClient` class from PRD Section 6.1
- [ ] **3.2** Implement `get_video_metadata(video_id: str) -> dict` method
- [ ] **3.3** Implement `get_video_caption(video_id: str) -> str` method
- [ ] **3.4** Add connection lifecycle methods (`__aenter__`, `__aexit__` or explicit `close()`)
- [ ] **3.5** Add exponential backoff retry logic (4 attempts, 2s base delay)
- [ ] **3.6** Create `app/services/__init__.py`

**Verification**:
1. Unit test with `respx` mocking HTTP responses
2. Test 404 handling and retry logic

**Note**: Connection to DB is probably not working yet, so for the moment avoid the verifications.

---

## Phase 4: RabbitMQ Consumer

**Goal**: Set up message consumer to listen for `transcript.fetched` events.

- [ ] **4.1** Create `app/consumers/video_processor.py` with consumer class from PRD Section 6.2
- [ ] **4.2** Implement RabbitMQ connection using `aio-pika` (connection, channel, queue binding)
- [ ] **4.3** Implement message handler skeleton (receives message, logs video_id)
- [ ] **4.4** Add duplicate check logic (query VideoDocument by source_video_id)
- [ ] **4.5** Create `app/consumers/__init__.py`
- [ ] **4.6** Add consumer startup to FastAPI lifespan events in `app/main.py`
- [ ] **4.7** Implement dead-letter queue (DLQ) publishing for failed messages

**Verification**:
1. Publish test message to RabbitMQ queue
2. Verify consumer logs the video_id
3. Verify duplicate messages are skipped

**Note**: The project infrastructure will be created with docker-compose, so for the moment avoid the verifications.

---

## Phase 5: AI Services

**Goal**: Implement LLM-based formatting, token-based chunking, and embedding generation.

### 5A: Formatting Service (AI Agent 1)
- [ ] **5.1** Create `app/services/formatting_service.py` from PRD Section 6.3
- [ ] **5.2** Configure `ChatOpenAI` with OpenRouter settings (api_key, base_url, model)
- [ ] **5.3** Implement formatting prompt template (NO SUMMARIZATION, add paragraph breaks, fix punctuation)
- [ ] **5.4** Add 60-second timeout and retry logic (max 3 retries on rate limits)
- [ ] **5.5** Write unit test with mocked LLM response

### 5B: Chunking Service
- [ ] **5.6** Create `app/services/chunking_service.py` from PRD Section 6.4
- [ ] **5.7** Implement `RecursiveCharacterTextSplitter` with tiktoken length function
- [ ] **5.8** Configure: chunk_size=400 tokens, overlap=50 tokens, separators=["\n\n", "\n", ". ", " "]
- [ ] **5.9** Return list of `(chunk_index, content)` tuples
- [ ] **5.10** Write unit test with sample text

### 5C: Embedding Service
- [ ] **5.11** Create `app/services/embedding_service.py` from PRD Section 6.5
- [ ] **5.12** Configure `OpenAIEmbeddings` with OpenRouter for `baai/bge-base-en-v1.5`
- [ ] **5.13** Implement `embed_documents(chunks: List[str]) -> List[List[float]]`
- [ ] **5.14** Validate embedding dimensions (must be 768)
- [ ] **5.15** Add retry logic for rate limits
- [ ] **5.16** Write unit test with mocked embedding response

**Verification**:
1. Each service has passing unit tests
2. Formatting service transforms raw text
3. Chunking produces expected number of chunks
4. Embeddings are 768-dimensional

**Note**: For the moment skip the verifications.

---

## Phase 6: Video Processing Pipeline (Integration)

**Goal**: Wire up all services in the RabbitMQ consumer to complete the ingestion pipeline.

- [ ] **6.1** In `video_processor.py`, fetch video data using `FetchingServiceClient`
- [ ] **6.2** Create `VideoDocument` record (status=PENDING)
- [ ] **6.3** Call `FormattingService` → update `formatted_content` and status=PROCESSING
- [ ] **6.4** Call `ChunkingService` → get chunk list
- [ ] **6.5** Call `EmbeddingService` → get embedding vectors
- [ ] **6.6** Create `VideoChunk` records with embeddings
- [ ] **6.7** Update `VideoDocument` status=COMPLETED
- [ ] **6.8** Implement error handling: update status=FAILED, store error_message, publish to DLQ
- [ ] **6.9** Write integration test for full pipeline (mock external APIs)

**Verification**:
1. Publish message → VideoDocument created with status=COMPLETED
2. VideoChunks exist with embeddings
3. Error cases result in status=FAILED

**Note**: For the moment skip the verifications.

---

## Phase 7: RAG Service (Query Guardrail + Search)

**Goal**: Implement query validation/transformation and vector search.

### 7A: Query Guardrail (AI Agent 2)
- [ ] **7.1** Create `app/services/rag_service.py` from PRD Section 6.6
- [ ] **7.2** Implement query guardrail prompt template (GARBAGE→INVALID, INCOMPLETE→error, VAGUE→rewrite, GOOD→pass)
- [ ] **7.3** Parse LLM JSON response to extract status/transformed_query/error_message
- [ ] **7.4** Write unit tests for each guardrail case (garbage, incomplete, vague, good)

### 7B: Vector Search
- [ ] **7.5** Implement query embedding (same embedding service)
- [ ] **7.6** Implement PGVector cosine similarity search (see PRD Section 6.6 Step 3 SQL)
- [ ] **7.7** Add similarity threshold filtering (default 0.7, meaning distance < 0.3)
- [ ] **7.8** Add optional channel_id filtering
- [ ] **7.9** Implement context fetching (get chunk_index ± 1 neighbors)
- [ ] **7.10** Group results by video_id, extract metadata
- [ ] **7.11** Write integration test with test data in DB

**Verification**:
1. Garbage query returns INVALID error
2. Vague query is transformed to semantic sentence
3. Vector search returns relevant chunks
4. Context includes neighboring segments

**Note**: For the moment skip the verifications.

---

## Phase 8: API Endpoints

**Goal**: Implement REST API endpoints for search and monitoring.

- [ ] **8.1** Create `app/api/v1/schemas/search.py` with request/response schemas from PRD Section 8
- [ ] **8.2** Create `app/api/v1/schemas/__init__.py`
- [ ] **8.3** Create `app/api/v1/endpoints/search.py` with `POST /api/v1/search` endpoint
- [ ] **8.4** Create `app/api/v1/endpoints/stats.py` with `GET /api/v1/stats` endpoint
- [ ] **8.5** Create `app/api/v1/endpoints/__init__.py`
- [ ] **8.6** Create `app/api/v1/__init__.py`
- [ ] **8.7** Create `app/api/router.py` aggregating all v1 routes
- [ ] **8.8** Create `app/api/deps.py` with dependency injection (db session, services)
- [ ] **8.9** Create `app/api/__init__.py`
- [ ] **8.10** Create `app/main.py` with FastAPI app, lifespan events, router registration
- [ ] **8.11** Add `/health` and `/ready` endpoints
- [ ] **8.12** Write integration tests for search endpoint

**Verification**:
1. `GET /health` returns 200
2. `GET /ready` returns 200 when DB/RabbitMQ connected
3. `POST /api/v1/search` with valid query returns results
4. `GET /api/v1/stats` returns document counts

**Note**: For the moment skip the verifications.

---

## Phase 9: Testing Infrastructure

**Goal**: Set up comprehensive test fixtures and factories.

- [ ] **9.1** Create `tests/__init__.py`
- [ ] **9.2** Create `tests/conftest.py` with pytest fixtures (async session, testcontainers for PG/RabbitMQ)
- [ ] **9.3** Create `tests/factories/__init__.py`
- [ ] **9.4** Create `tests/factories/video_models.py` with polyfactory factories for VideoDocument/Chunk
- [ ] **9.5** Create `tests/unit/__init__.py`
- [ ] **9.6** Create `tests/integration/__init__.py`
- [ ] **9.7** Create `tests/e2e/__init__.py`
- [ ] **9.8** Ensure all unit tests pass with mocks
- [ ] **9.9** Ensure integration tests pass with testcontainers
- [ ] **9.10** Run coverage report: `pytest --cov=app --cov-report=html`

**Verification**:
1. `pytest tests/unit/` passes
2. `pytest tests/integration/` passes
3. Coverage > 80%

**Note**: For the moment skip the verifications.

---

## Phase 10: Docker & Final Integration

**Goal**: Verify Docker build and full system integration.

- [x] **10.1** Verify Dockerfile uses Python 3.12-slim and `app/` folder structure
- [x] **10.2** Verify docker-compose.yml includes ai-service with correct env_file
- [ ] **10.3** Build Docker image: `docker build -t ai-service .`
- [ ] **10.4** Run with docker-compose: `docker-compose up`
- [ ] **10.5** Verify health endpoint responds
- [ ] **10.6** Verify RabbitMQ consumer connects and listens
- [ ] **10.7** End-to-end test: publish message → verify processing → search for content
- [ ] **10.8** Clean up old files (`base.py`, `video_models.py` at root level) after confirming migration

**Verification**:
1. `docker-compose up` starts all services
2. AI service connects to RabbitMQ and PostgreSQL
3. Full E2E flow works

**Note**: For the moment skip the verifications.

---

## Phase Dependencies

```
Phase 1 (Setup) 
    ↓
Phase 2 (Database) 
    ↓
Phase 3 (HTTP Client) ──────┐
    ↓                       │
Phase 4 (RabbitMQ Consumer) │
    ↓                       │
Phase 5 (AI Services)       │
    ↓                       │
Phase 6 (Pipeline Integration) ←──┘
    ↓
Phase 7 (RAG Service)
    ↓
Phase 8 (API Endpoints)
    ↓
Phase 9 (Testing)
    ↓
Phase 10 (Docker & Final)
```

---

## Notes for AI Agent

1. **Virtual Environment**: The virtual environment is managed by virtualenvwrapper, named `yt-scraper`. Use `workon yt-scraper` to activate.

2. **External API Mocking**: When testing services that call OpenRouter (formatting, embedding, guardrail), always use `respx` to mock HTTP calls. Real API calls cost money.

3. **Database**: Use `pgvector/pgvector:pg15` Docker image which has PGVector pre-installed.

4. **Existing Files**: The following files exist at the root of `ai-service/` and need to be relocated in Phase 2:
   - `base.py` → `app/models/base.py`
   - `video_models.py` → `app/models/videos.py`

5. **Configuration Loading**: Use `pydantic-settings` to load from `.env` file. The Settings class is a singleton.

6. **Async Everything**: All database operations, HTTP calls, and RabbitMQ operations must be async. Use `asyncpg` for PostgreSQL, `httpx` for HTTP, `aio-pika` for RabbitMQ.

7. **Error Handling**: Follow PRD Section 10 for retry strategies and error response formats.

---

**END OF IMPLEMENTATION CHECKLIST**
