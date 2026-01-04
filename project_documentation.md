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
| `POST` | `/api/v1/process-video-by-id` | Manually trigger processing for a specific source_video_id (testing/debugging) |
| `GET` | `/api/v1/stats` | Get service statistics (processed count, pending count, failed count, total chunks) |

### Health & Readiness Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check endpoint |
| `GET` | `/ready` | Readiness check (DB and RabbitMQ connection validation) |

---

## Architecture & Key Features

### Authentication System
- **Email-based login**: Primary authentication via email and password
- **JWT tokens**: Access tokens (30min) + refresh tokens (7 days)
- **Optional authentication**: Search endpoint works for authenticated and anonymous users
- **Search history tracking**: Automatic logging of authenticated user searches
- **Secure passwords**: Bcrypt hashing with strength validation

### Message-Driven Processing
- **Event ingestion**: RabbitMQ consumer listening to `transcript.fetched` queue
- **AI processing**: LLM-based transcript formatting → chunking → embedding generation
- **Vector storage**: PostgreSQL with PGVector for semantic search
- **Retry logic**: Exponential backoff with dead-letter queue for failures

### Core Components
- **RAGService**: Semantic search with query validation and context assembly
- **FormattingService**: LLM-based transcript cleaning
- **EmbeddingService**: 768-dimensional vector generation
- **ChunkingService**: Token-aware text splitting (400 tokens/chunk)

### Technology Stack
- FastAPI (async web framework)
- PostgreSQL + PGVector (vector database)
- RabbitMQ (message broker)
- LangChain (AI orchestration)
- OpenRouter (LLM/embeddings provider)
- python-jose + passlib (authentication)

---

## Configuration

Environment variables required:
- `DATABASE_URL`: PostgreSQL connection string
- `RABBITMQ_URL`: RabbitMQ connection URL
- `OPENROUTER_API_KEY`: OpenRouter API key
- `FETCHING_SERVICE_URL`: URL to fetching service
- `JWT_SECRET_KEY`: JWT signing secret (generate with `openssl rand -hex 32`)
- `JWT_ALGORITHM`: JWT algorithm (default: HS256)
- `LOG_LEVEL`: Logging level (default: INFO)

See `sample.env` for complete configuration options and `AUTHENTICATION.md` for auth setup.

