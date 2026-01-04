# AI Video Processing Service - Project Documentation

## Project Overview

A FastAPI-based microservice that processes raw video transcripts, formats them into readable articles, creates vector embeddings, and provides a RAG-based search API. The service ingests data from RabbitMQ events, orchestrates AI agents for text processing, and stores formatted content and embeddings in PostgreSQL with PGVector for semantic search.

---

## API Endpoints

### Core Search & Processing Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/search` | Perform RAG query with optional filtering (query, channel_id, top_k, similarity_threshold) |
| `POST` | `/api/v1/process-video-by-id` | Manually trigger processing for a specific source_video_id (testing/debugging) |
| `GET` | `/api/v1/stats` | Get service statistics (processed count, pending count, failed count, total chunks) |

### Health & Readiness Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check endpoint |
| `GET` | `/ready` | Readiness check (DB and RabbitMQ connection validation) |
