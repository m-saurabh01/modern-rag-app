# Modern RAG Application

## Overview
Industry-standard RAG (Retrieval-Augmented Generation) application with modern embeddings, semantic chunking, and production-ready architecture.

## Features
- Modern embedding-based document processing
- Semantic chunking with overlap preservation
- Async FastAPI architecture
- Production-grade error handling and logging
- Scalable document processing pipeline
- Advanced RAG techniques
- **Optional streaming responses** for real-time interaction
- LLaMA integration with intelligent fallbacks
- Multi-modal retrieval with adaptive weighting
- ChromaDB and Qdrant vector store support

## Requirements
- Python 3.9+
- 32GB RAM (optimized for CPU-only processing)
- SSD storage for vector database
- Offline-capable (all dependencies portable)

## Architecture
```
├── app/                    # Main application
├── services/              # Business logic services  
├── core/                  # Configuration and utilities
├── api/                   # FastAPI routes
├── models/                # Data models and schemas
└── processing/            # Document processing pipeline
```

## Development Status
🚧 **In Development** - Following industry-standard practices

## Quick Start

### Basic Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python api/v1/endpoints.py

# Access documentation
http://localhost:8000/docs
```

### Testing Features
```bash
# Test standard responses
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the budget?", "mode": "balanced"}'

# Test streaming responses (optional)
curl -X POST "http://localhost:8000/ask/stream" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the budget?", "mode": "balanced"}'

# Run streaming demo
python demo_streaming.py
```
