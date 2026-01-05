# AI Video Processing Service - Project Documentation

## Project Overview

A FastAPI-based microservice that processes raw video transcripts, formats them into readable articles, creates vector embeddings, and provides a RAG-based search API. The service ingests data from RabbitMQ events, orchestrates AI agents for text processing, and stores formatted content and embeddings in PostgreSQL with PGVector for semantic search.

---

## API Endpoints

### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/auth/register` | Register new user (email, password, optional phone/name) |
| `POST` | `/api/v1/auth/login` | Login with email and password, returns JWT tokens |
| `POST` | `/api/v1/auth/refresh` | Refresh access token using refresh token |
| `GET` | `/api/v1/auth/me` | Get current user profile (requires authentication) |
| `PATCH` | `/api/v1/auth/me` | Update user profile (requires authentication) |
| `GET` | `/api/v1/auth/me/search-history` | Get user's search history with pagination (requires authentication) |

### Core Search & Processing Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/search` | Perform RAG query with optional authentication (query, channel_id, top_k, similarity_threshold). If authenticated, search is tracked in history |
| `GET` | `/api/v1/videos` | List all processed videos with pagination (skip, limit). Returns video metadata and processing status |
| `POST` | `/api/v1/process-video-by-id` | Manually trigger processing for a specific source_video_id (testing/debugging) |
| `GET` | `/api/v1/stats` | Get service statistics (processed count, pending count, failed count, total chunks) |

### Health & Readiness Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check endpoint |
| `GET` | `/ready` | Readiness check (DB and RabbitMQ connection validation) |

---

## Architecture Overview

### Message-Driven Processing Pipeline

The service uses **event-driven architecture** with RabbitMQ:

1. **Ingestion**: Listens to `transcript.fetched` queue for messages containing video_id
2. **Data Fetching**: Retrieves raw transcript and metadata from Fetching Service API (`GET /videos/{video_id}` and `GET /videos/{video_id}/caption`)
3. **AI Processing**:
   - **Formatting Agent** (AI Agent 1): Cleans and formats raw transcript using OpenRouter's gpt-oss-120b
   - **Chunking**: Splits formatted text into 400-token chunks using recursive character splitting
   - **Embedding Generation**: Creates 768-dimensional vector embeddings via OpenRouter's baai/bge-base-en-v1.5
4. **Persistence**: Stores formatted content and embeddings in PostgreSQL with PGVector
5. **RAG Search**: 
   - **Query Validation Agent** (AI Agent 2): Validates and transforms user queries
   - **Semantic Search**: Uses cosine similarity to find relevant chunks
   - **Context Assembly**: Returns matching chunks with neighboring context

### Core Services

| Service | File | Purpose |
|---------|------|---------|
| `FetchingServiceClient` | `app/services/fetching_client.py` | HTTP wrapper for Fetching Service API |
| `FormattingService` | `app/services/formatting_service.py` | LLM-based transcript formatting |
| `ChunkingService` | `app/services/chunking_service.py` | Token-aware text splitting (400 tokens/chunk) |
| `EmbeddingService` | `app/services/embedding_service.py` | Vector embedding generation |
| `RAGService` | `app/services/rag_service.py` | RAG queries with query guardrails |
| `VideoProcessor` | `app/consumers/video_processor.py` | RabbitMQ message consumer for processing |

---

## Key Libraries & Technologies

| Component | Technology | Version | Notes |
|-----------|-----------|---------|-------|
| **Framework** | FastAPI | >=0.115.0 | Asynchronous web framework |
| **Python** | Python | 3.12 | |
| **Database** | PostgreSQL + PGVector | 15+ | Vector-capable database for embeddings |
| **ORM** | SQLAlchemy (async) | >=2.0.0 | Async database interaction |
| **Migrations** | Alembic | latest | Schema management |
| **Messaging** | RabbitMQ + aio-pika | 3-mgmt / >=9.4.0 | Inter-service event communication |
| **AI Orchestration** | LangChain / LangGraph | >=0.3.0 | AI agent and RAG pipeline framework |
| **LLM Access** | langchain-openai | >=0.2.0 | OpenAI-compatible API interface (OpenRouter) |
| **HTTP Client** | httpx | >=0.28.0 | Async HTTP calls to Fetching Service |
| **Token Counting** | tiktoken | >=0.7.0 | Accurate token-based chunk sizing |
| **Authentication** | python-jose | >=3.3.0 | JWT token generation and validation |
| **Password Hashing** | passlib[bcrypt] | >=1.7.4 | Secure password hashing with bcrypt |

### AI Models (via OpenRouter)

| Task | Model | Dimensions | Purpose |
|------|-------|-----------|---------|
| **Formatting** | openai/gpt-oss-120b | N/A | Clean and format raw transcripts (Agent 1) |
| **Embeddings** | baai/bge-base-en-v1.5 | 768 | Generate semantic vector embeddings |
| **Query Processing** | openai/gpt-oss-120b | N/A | Validate and transform user queries (Agent 2) |

---

## Data Models

### Database Schema

**User** (Authentication)
- `id`: UUID primary key
- `email`: Email address (unique, primary identifier)
- `hashed_password`: Bcrypt hashed password
- `phone`: Phone number (optional, unique if provided)
- `full_name`: Full name (optional)
- `is_active`: Account active status
- `is_superuser`: Admin privileges flag
- `created_at`, `updated_at`: Timestamps

**SearchHistory** (User activity tracking)
- `id`: UUID primary key
- `user_id`: FK to User
- `query`: Original search query
- `transformed_query`: AI-transformed query
- `channel_id`: Optional channel filter
- `top_k`: Number of results requested
- `similarity_threshold`: Similarity threshold used
- `results_count`: Number of results returned
- `processing_time_ms`: Processing time in milliseconds
- `created_at`: Timestamp

**VideoDocument** (Parent model)
- `id`: UUID primary key
- `source_video_id`: YouTube video ID (unique)
- `source_channel_id`: YouTube channel ID
- `formatted_content`: Processed transcript text
- `status`: PENDING | PROCESSING | COMPLETED | FAILED
- `error_message`: Error details if status is FAILED
- `meta_data`: JSON with title, channel_name, published_at, duration_seconds
- `created_at`, `updated_at`: Timestamps

**VideoChunk** (Child model)
- `id`: UUID primary key
- `document_id`: FK to VideoDocument
- `chunk_index`: Sequential position in document
- `content`: Chunk text
- `embedding`: 768-dimensional vector (PGVector)
- `created_at`: Timestamp

---

## Request/Response Examples

### Search Request
```json
{
  "query": "What strategies does the speaker discuss for success?",
  "channel_id": null,
  "top_k": 5,
  "similarity_threshold": 0.7
}
```

### Search Response
```json
{
  "query": "What strategies does the speaker discuss for success?",
  "transformed_query": "What strategies does the speaker discuss for success?",
  "results": [
    {
      "video_id": "abc123",
      "title": "Success Strategies Interview",
      "channel_name": "Channel Name",
      "published_at": "2024-01-15T10:30:00Z",
      "matched_segment": "In my experience, three key strategies...",
      "previous_segment": "Let me start with the foundation...",
      "next_segment": "These principles apply in most situations..."
    }
  ],
  "total_results": 1,
  "processing_time_ms": 234
}
```

---

## Error Handling

### RabbitMQ Consumer Resilience
- **Max Retries**: 4 with exponential backoff (2s, 4s, 8s, 16s)
- **Dead Letter Queue**: Failed messages sent to `transcript.failed` queue for manual review
- **Non-Retryable Errors**: 404, validation errors, malformed messages

### API Error Responses
- Invalid queries (garbage, incomplete): 400 Bad Request
- Service errors: 500 Internal Server Error
- Timeouts: 504 Gateway Timeout

---

## Configuration

Key environment variables:
- `FETCHING_SERVICE_URL`: URL to Fetching Service API (default: `http://fetching-service:8000`)
- `OPENROUTER_API_KEY`: API key for OpenRouter
- `DATABASE_URL`: PostgreSQL connection string with PGVector support
- `RABBITMQ_URL`: RabbitMQ connection URL
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `JWT_SECRET_KEY`: Secret key for JWT token signing (required for authentication)
- `JWT_ALGORITHM`: JWT signing algorithm (default: `HS256`)
- `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`: Access token expiration in minutes (default: `30`)
- `JWT_REFRESH_TOKEN_EXPIRE_DAYS`: Refresh token expiration in days (default: `7`)

---

## Authentication

The service includes JWT-based authentication with the following features:
- **Email-based login**: Users authenticate with email and password
- **Optional authentication**: Search endpoint works for both authenticated and anonymous users
- **Automatic history tracking**: Authenticated searches are automatically logged
- **Token-based auth**: Access tokens (30min) and refresh tokens (7 days)
- **Secure passwords**: Bcrypt hashing with strength requirements (8+ chars, uppercase, lowercase, digit)

---
