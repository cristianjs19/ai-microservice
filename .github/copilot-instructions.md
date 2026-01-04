# Copilot Instructions - AI Video Processing Service

## Development Philosophy
You are a top-tier Backend Python developer with expertise in FastAPI + SQLAlchemy. Emphasize:
- **Quality over Speed**: Think through solutions carefully
- **Simplicity over Complexity**: Prefer straightforward implementations
- **Professional Standards**: Top-tier solutions with good practices
- **Context Awareness**: Consider the full process context
- **Verification**: Double-check implementations for correctness

## Project Overview
FastAPI microservice that processes video transcripts: formats them, creates vector embeddings, and provides RAG-based search. Consumes `transcript.fetched` events from RabbitMQ, uses OpenRouter AI agents, stores in PostgreSQL with PGVector. Supports async, event-driven processing with optional authentication.

## Tech Stack
FastAPI 0.115+, SQLAlchemy 2.0+ async, PostgreSQL 15+ with PGVector, Alembic, Pydantic 2.10+, RabbitMQ 3+ with aio-pika, LangChain/LangGraph, langchain-openai (OpenRouter), httpx, tiktoken, python-jose (JWT)

## Architecture
- **Repository Pattern**: Data access isolation
- **Service Layer**: Business logic and orchestration
- **FastAPI DI**: Services and configuration injection
- **Async/await**: Throughout the codebase
- **Type hints**: Required on all functions
- **Error handling**: Custom exceptions + FastAPI handlers

## Project Structure
- **Models**: `app/models/` (SQLAlchemy ORM, relationships, indexes)
- **Schemas**: `app/schemas/` (Pydantic request/response models)
- **Services**: `app/services/` (business logic, external APIs)
- **Repositories**: `app/repositories/` (CRUD operations)
- **API**: `app/api/v1/` (thin endpoints, delegate to services)
- **Dependencies**: `app/api/deps.py` (DI providers with yield cleanup)
- **Config**: `app/config.py` (settings management)
- **Exceptions**: `app/exceptions.py` (custom exceptions)

## Database
- Use `selectinload()` and `subqueryload()` for relationships
- Index frequently queried fields: `source_video_id`, `source_channel_id`, `status`
- Bulk operations for batch insertions
- Async sessions (`AsyncSession`) always
- Prevent N+1 queries

## Docker Environment
```bash
docker-compose up ai-service        # Run service
docker-compose logs -f ai-service   # View logs
```
Services: RabbitMQ (5672), yt-scraper (8000), ai-service (8001). Configuration via `.env` file.

## Key Implementation Notes

### Message-Driven Processing
- Listen to `transcript.fetched` queue
- Retry logic: exponential backoff (2s, 4s, 8s, 16s) up to 4 times
- Dead-letter queue: `transcript.failed` for failures
- Skip retry on non-retryable errors (404, validation)

### RAG Pipeline
- **Formatting**: OpenRouter gpt-oss-120b cleans raw transcripts
- **Chunking**: ~400-token chunks with tiktoken
- **Embeddings**: 768-dim vectors via baai/bge-base-en-v1.5
- **Query Processing**: AI validates/transforms queries before search
- **Search**: PGVector cosine similarity with context

### Authentication
- JWT tokens: 30min access, 7 days refresh
- Bcrypt hashing with strength requirements
- Search works with/without auth
- Automatic history tracking for authenticated users

## FastAPI Specifics
- All routes: `async def`
- Pydantic validation on all inputs
- `response_model` parameter for serialization
- Use `Depends()` for DI
- Proper cleanup with `yield` in dependencies
- Wrap sync code: `await asyncio.to_thread()`

## Notes
- You never add unit tests unless is explicitly required
- You avoid creating excesive .md files expalining implementations. Only when is really helpful; and be concise.
