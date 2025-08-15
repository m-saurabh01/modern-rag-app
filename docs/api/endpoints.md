# API Endpoints Specification

Detailed specifications for all REST API endpoints in the Modern RAG Application.

## 📋 Table of Contents

- [Health & System Endpoints](#health--system-endpoints)
- [Document Management Endpoints](#document-management-endpoints)
- [Query Processing Endpoints](#query-processing-endpoints)
- [Search Endpoints](#search-endpoints)
- [Configuration Endpoints](#configuration-endpoints)
- [Collection Management](#collection-management)
- [WebSocket Endpoints](#websocket-endpoints)

---

## 🏥 Health & System Endpoints

### GET `/health`
**Description:** Basic health check endpoint for monitoring services.

**Parameters:** None

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-08-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "vector_db": "operational",
    "embedding_service": "operational", 
    "llm_service": "operational"
  }
}
```

**Status Codes:**
- `200 OK`: All services healthy
- `503 Service Unavailable`: One or more services unavailable

---

### GET `/status`
**Description:** Detailed system status and performance metrics.

**Parameters:** None

**Response:**
```json
{
  "system": {
    "status": "operational",
    "uptime": 7200,
    "load_average": [1.2, 1.5, 1.8],
    "memory": {
      "used": 4096,
      "available": 12288,
      "usage_percent": 25.0
    }
  },
  "services": {
    "vector_database": {
      "status": "healthy",
      "connections": 5,
      "response_time_ms": 12
    },
    "embedding_service": {
      "status": "healthy",
      "model_loaded": true,
      "cache_hit_rate": 0.85
    },
    "llm_service": {
      "status": "healthy",
      "model": "llama3",
      "queue_size": 0
    }
  },
  "statistics": {
    "total_queries": 1500,
    "successful_queries": 1485,
    "average_response_time": 0.85,
    "total_documents": 50,
    "total_embeddings": 750
  }
}
```

---

### GET `/metrics`
**Description:** Prometheus-compatible metrics for monitoring.

**Parameters:**
- `format` (query, optional): Response format (`json`, `prometheus`)

**Response (JSON format):**
```json
{
  "http_requests_total": {
    "type": "counter",
    "value": 1500,
    "labels": {"method": "POST", "endpoint": "/query"}
  },
  "response_time_seconds": {
    "type": "histogram",
    "buckets": {
      "0.1": 100,
      "0.5": 800,
      "1.0": 1400,
      "2.0": 1500
    }
  },
  "active_connections": {
    "type": "gauge",
    "value": 25
  }
}
```

---

## 📄 Document Management Endpoints

### POST `/documents/upload`
**Description:** Upload and process documents for vector storage.

**Content-Type:** `multipart/form-data`

**Parameters:**
- `files` (form-data, required): One or more files to upload
- `collection_name` (form-data, optional): Target collection name
- `processing_mode` (form-data, optional): `fast`, `balanced`, `thorough`
- `extract_metadata` (form-data, optional): Extract document metadata
- `chunk_size` (form-data, optional): Custom chunk size
- `chunk_overlap` (form-data, optional): Chunk overlap percentage

**Request Example:**
```bash
curl -X POST "http://localhost:8000/documents/upload" \
     -F "files=@document1.pdf" \
     -F "files=@document2.pdf" \
     -F "collection_name=research_papers" \
     -F "processing_mode=balanced" \
     -F "extract_metadata=true"
```

**Response:**
```json
{
  "upload_id": "upload_12345",
  "status": "completed",
  "uploaded_documents": [
    {
      "filename": "document1.pdf",
      "document_id": "doc_001",
      "status": "processed",
      "file_size": 2048576,
      "pages": 10,
      "chunks_created": 15,
      "embeddings_generated": 15,
      "processing_time": 3.45,
      "metadata": {
        "title": "Research Paper on ML",
        "author": "John Doe",
        "creation_date": "2024-01-15",
        "language": "en"
      }
    }
  ],
  "failed_documents": [],
  "statistics": {
    "total_files": 2,
    "successful": 2,
    "failed": 0,
    "total_processing_time": 6.89,
    "total_chunks": 28,
    "total_embeddings": 28
  }
}
```

**Status Codes:**
- `201 Created`: Upload successful
- `400 Bad Request`: Invalid file format or parameters
- `413 Payload Too Large`: File size exceeds limit
- `422 Unprocessable Entity`: Processing error

---

### GET `/documents`
**Description:** List all uploaded documents with pagination and filtering.

**Parameters:**
- `collection_name` (query, optional): Filter by collection
- `limit` (query, optional): Number of results per page (default: 50)
- `offset` (query, optional): Pagination offset (default: 0)
- `status` (query, optional): Filter by processing status
- `sort_by` (query, optional): Sort field (`upload_date`, `filename`, `size`)
- `sort_order` (query, optional): Sort order (`asc`, `desc`)

**Response:**
```json
{
  "documents": [
    {
      "document_id": "doc_001",
      "filename": "research_paper.pdf",
      "collection_name": "research_papers",
      "status": "processed",
      "upload_date": "2024-08-15T10:30:00Z",
      "file_size": 2048576,
      "chunks": 15,
      "metadata": {
        "pages": 10,
        "content_type": "application/pdf",
        "language": "en"
      }
    }
  ],
  "pagination": {
    "total": 50,
    "limit": 50,
    "offset": 0,
    "has_next": false,
    "has_previous": false
  },
  "filters_applied": {
    "collection_name": "research_papers",
    "status": "processed"
  }
}
```

---

### GET `/documents/{document_id}`
**Description:** Get detailed information about a specific document.

**Path Parameters:**
- `document_id` (required): Unique document identifier

**Response:**
```json
{
  "document_id": "doc_001",
  "filename": "research_paper.pdf",
  "collection_name": "research_papers",
  "status": "processed",
  "upload_date": "2024-08-15T10:30:00Z",
  "last_modified": "2024-08-15T10:33:45Z",
  "file_info": {
    "size": 2048576,
    "content_type": "application/pdf",
    "md5_hash": "d41d8cd98f00b204e9800998ecf8427e"
  },
  "processing_info": {
    "mode": "balanced",
    "processing_time": 3.45,
    "chunks_created": 15,
    "embeddings_generated": 15,
    "extraction_method": "pdf_processor"
  },
  "metadata": {
    "title": "Research Paper on Machine Learning",
    "author": "John Doe",
    "subject": "Machine Learning",
    "creation_date": "2024-01-15",
    "pages": 10,
    "language": "en",
    "keywords": ["machine learning", "neural networks"]
  },
  "statistics": {
    "query_count": 25,
    "last_queried": "2024-08-15T15:30:00Z",
    "average_relevance_score": 0.78
  }
}
```

**Status Codes:**
- `200 OK`: Document found
- `404 Not Found`: Document doesn't exist

---

### GET `/documents/{document_id}/chunks`
**Description:** Get all text chunks for a specific document.

**Path Parameters:**
- `document_id` (required): Unique document identifier

**Query Parameters:**
- `limit` (optional): Maximum chunks to return (default: 100)
- `offset` (optional): Pagination offset
- `include_embeddings` (optional): Include embedding vectors

**Response:**
```json
{
  "document_id": "doc_001",
  "chunks": [
    {
      "chunk_id": "chunk_001",
      "content": "This research paper explores the applications of machine learning in modern software development...",
      "metadata": {
        "page": 1,
        "paragraph": 1,
        "start_char": 0,
        "end_char": 150,
        "chunk_index": 0
      },
      "embedding": [0.1, -0.2, 0.3, ...], // Only if include_embeddings=true
      "statistics": {
        "character_count": 150,
        "word_count": 25,
        "sentence_count": 2
      }
    }
  ],
  "total_chunks": 15,
  "pagination": {
    "limit": 100,
    "offset": 0,
    "has_next": false
  }
}
```

---

### DELETE `/documents/{document_id}`
**Description:** Delete a document and all associated data.

**Path Parameters:**
- `document_id` (required): Unique document identifier

**Query Parameters:**
- `remove_embeddings` (optional): Remove from vector database (default: true)
- `remove_files` (optional): Remove stored files (default: true)

**Response:**
```json
{
  "document_id": "doc_001",
  "filename": "research_paper.pdf",
  "status": "deleted",
  "deleted_components": {
    "document_record": true,
    "text_chunks": 15,
    "embeddings": 15,
    "stored_files": true,
    "metadata": true
  },
  "deletion_time": 0.45
}
```

**Status Codes:**
- `200 OK`: Document deleted successfully
- `404 Not Found`: Document doesn't exist
- `409 Conflict`: Document is being processed

---

## 🔍 Query Processing Endpoints

### POST `/query`
**Description:** Process a natural language query through the complete RAG pipeline.

**Request Body:**
```json
{
  "query": "What are the main applications of machine learning?",
  "collection_name": "research_papers",
  "mode": "balanced",
  "response_format": "with_sources",
  "max_results": 10,
  "score_threshold": 0.7,
  "include_analysis": true,
  "include_metadata": true,
  "stream": false,
  "language": "en",
  "context": {
    "previous_queries": [],
    "user_preferences": {}
  },
  "filters": {
    "document_types": ["pdf"],
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    }
  }
}
```

**Response:**
```json
{
  "query_id": "query_12345",
  "query": "What are the main applications of machine learning?",
  "response": "Machine learning has several key applications including: 1) Predictive Analytics - used in finance and marketing for forecasting trends...",
  "sources": [
    {
      "content": "Machine learning applications span across various industries...",
      "metadata": {
        "document_id": "doc_001",
        "source": "ml_applications.pdf",
        "page": 5,
        "chunk_id": "chunk_045"
      },
      "relevance_score": 0.95,
      "citation_number": 1
    }
  ],
  "query_analysis": {
    "intent": "informational",
    "entities": ["machine learning", "applications"],
    "confidence": 0.92,
    "query_type": "factual_question",
    "complexity": "medium"
  },
  "processing_metadata": {
    "mode": "balanced",
    "retrieval_time": 0.15,
    "generation_time": 1.08,
    "total_time": 1.23,
    "tokens_generated": 156,
    "sources_considered": 25,
    "sources_used": 5
  },
  "collection_info": {
    "name": "research_papers",
    "documents_searched": 50,
    "chunks_searched": 750
  }
}
```

---

### POST `/query/stream`
**Description:** Process query with streaming response generation.

**Request Body:** Same as `/query` endpoint

**Response:** Server-Sent Events (SSE) stream
```
event: analysis
data: {"status": "analyzing_query", "intent": "informational", "entities": ["machine learning"]}

event: retrieval
data: {"status": "searching", "documents_found": 5, "best_score": 0.95}

event: generation_start
data: {"status": "generating_response", "estimated_tokens": 200}

event: token
data: {"content": "Machine"}

event: token  
data: {"content": " learning"}

event: citation
data: {"number": 1, "source": "ml_applications.pdf", "page": 5}

event: token
data: {"content": " has"}

event: complete
data: {"total_time": 2.1, "tokens_generated": 156, "sources_used": 5}
```

---

### POST `/analyze`
**Description:** Analyze query without document retrieval or response generation.

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "include_entities": true,
  "include_intent": true,
  "include_complexity": true,
  "language": "en"
}
```

**Response:**
```json
{
  "query": "What is machine learning?",
  "analysis": {
    "intent": {
      "primary": "definitional",
      "confidence": 0.95,
      "secondary_intents": ["informational"]
    },
    "entities": [
      {
        "text": "machine learning",
        "label": "TECHNOLOGY",
        "confidence": 0.98,
        "start": 8,
        "end": 24
      }
    ],
    "complexity": {
      "level": "simple",
      "factors": {
        "question_type": "what_is",
        "entity_count": 1,
        "ambiguity": "low"
      }
    },
    "language": {
      "detected": "en",
      "confidence": 0.99
    },
    "query_type": "factual_question",
    "suggested_mode": "fast",
    "processing_recommendations": {
      "expected_sources": 3,
      "response_length": "medium"
    }
  },
  "processing_time": 0.05
}
```

---

## 🔎 Search Endpoints

### POST `/search`
**Description:** Search documents without LLM response generation.

**Request Body:**
```json
{
  "query": "neural network architectures",
  "collection_name": "research_papers",
  "limit": 20,
  "offset": 0,
  "score_threshold": 0.7,
  "include_metadata": true,
  "search_type": "semantic",
  "filters": {
    "document_types": ["pdf"],
    "authors": ["John Doe"],
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    }
  },
  "rerank": true,
  "rerank_top_k": 10
}
```

**Response:**
```json
{
  "query": "neural network architectures",
  "results": [
    {
      "content": "Convolutional neural networks (CNNs) are a class of deep neural networks...",
      "metadata": {
        "document_id": "doc_003",
        "source": "cnn_architectures.pdf",
        "page": 12,
        "chunk_id": "chunk_156",
        "author": "Jane Smith",
        "creation_date": "2024-03-15"
      },
      "relevance_score": 0.94,
      "distance": 0.06,
      "rank": 1,
      "highlight": {
        "content": "...Convolutional **neural network** **architectures** are designed..."
      }
    }
  ],
  "search_metadata": {
    "total_results": 15,
    "search_time": 0.08,
    "rerank_time": 0.03,
    "filters_applied": 3,
    "search_type": "semantic"
  },
  "collection_info": {
    "name": "research_papers",
    "total_documents": 50,
    "total_chunks": 750,
    "documents_matched": 8
  },
  "pagination": {
    "limit": 20,
    "offset": 0,
    "has_next": false,
    "next_offset": null
  }
}
```

---

### POST `/search/similar`
**Description:** Find documents similar to a given document.

**Request Body:**
```json
{
  "document_id": "doc_001",
  "limit": 10,
  "score_threshold": 0.7,
  "collection_name": "research_papers",
  "exclude_self": true,
  "similarity_metric": "cosine"
}
```

**Response:**
```json
{
  "reference_document": {
    "document_id": "doc_001",
    "filename": "reference_paper.pdf",
    "title": "Introduction to Neural Networks"
  },
  "similar_documents": [
    {
      "document_id": "doc_005",
      "filename": "deep_learning_basics.pdf",
      "title": "Deep Learning Fundamentals",
      "similarity_score": 0.89,
      "common_topics": ["neural networks", "deep learning", "backpropagation"],
      "metadata": {
        "author": "Alice Johnson",
        "pages": 25,
        "creation_date": "2024-02-10"
      }
    }
  ],
  "search_metadata": {
    "similarity_metric": "cosine",
    "computation_time": 0.12,
    "compared_documents": 49
  }
}
```

---

## ⚙️ Configuration Endpoints

### GET `/config`
**Description:** Get current system configuration.

**Response:**
```json
{
  "vector_database": {
    "type": "chromadb",
    "host": "localhost",
    "port": 8000,
    "default_collection": "default_collection",
    "embedding_function": "all-MiniLM-L6-v2"
  },
  "embedding_service": {
    "model": "all-MiniLM-L6-v2",
    "dimension": 384,
    "cache_enabled": true,
    "batch_size": 32
  },
  "llm_service": {
    "provider": "ollama",
    "model": "llama3",
    "base_url": "http://localhost:11434",
    "context_length": 4096,
    "temperature": 0.7
  },
  "processing": {
    "default_mode": "balanced",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "supported_formats": ["pdf", "txt", "docx", "md"]
  },
  "performance": {
    "max_concurrent_requests": 50,
    "request_timeout": 300,
    "embedding_cache_size": 10000,
    "response_cache_ttl": 3600
  },
  "security": {
    "authentication_enabled": false,
    "rate_limiting_enabled": true,
    "cors_enabled": true,
    "allowed_origins": ["http://localhost:3000"]
  }
}
```

---

### PUT `/config`
**Description:** Update system configuration (requires admin privileges).

**Request Body:**
```json
{
  "llm_service": {
    "temperature": 0.8,
    "context_length": 8192
  },
  "processing": {
    "default_mode": "thorough",
    "chunk_size": 750
  },
  "performance": {
    "max_concurrent_requests": 100
  }
}
```

**Response:**
```json
{
  "status": "updated",
  "updated_settings": [
    "llm_service.temperature",
    "llm_service.context_length", 
    "processing.default_mode",
    "processing.chunk_size",
    "performance.max_concurrent_requests"
  ],
  "requires_restart": false,
  "effective_immediately": true,
  "previous_values": {
    "llm_service.temperature": 0.7,
    "processing.default_mode": "balanced"
  }
}
```

---

## 🗂️ Collection Management

### GET `/collections`
**Description:** List all vector database collections.

**Query Parameters:**
- `include_stats` (optional): Include collection statistics
- `include_empty` (optional): Include empty collections

**Response:**
```json
{
  "collections": [
    {
      "name": "research_papers",
      "created_date": "2024-08-01T10:00:00Z",
      "document_count": 50,
      "chunk_count": 750,
      "embedding_dimension": 384,
      "total_size_mb": 125.5,
      "last_modified": "2024-08-15T15:30:00Z",
      "metadata": {
        "description": "Academic research papers collection",
        "tags": ["academic", "research", "ml"]
      }
    }
  ],
  "total_collections": 3,
  "total_documents": 150,
  "total_chunks": 2250,
  "storage_info": {
    "total_size_mb": 456.7,
    "compression_ratio": 0.75
  }
}
```

---

### GET `/collections/{collection_name}/info`
**Description:** Get detailed information about a specific collection.

**Path Parameters:**
- `collection_name` (required): Name of the collection

**Response:**
```json
{
  "name": "research_papers",
  "created_date": "2024-08-01T10:00:00Z",
  "last_modified": "2024-08-15T15:30:00Z",
  "statistics": {
    "document_count": 50,
    "chunk_count": 750,
    "total_size_mb": 125.5,
    "average_chunk_size": 512,
    "embedding_dimension": 384
  },
  "documents": {
    "recent_uploads": [
      {
        "document_id": "doc_050",
        "filename": "latest_research.pdf",
        "upload_date": "2024-08-15T15:30:00Z",
        "chunks": 18
      }
    ],
    "most_queried": [
      {
        "document_id": "doc_001",
        "filename": "popular_paper.pdf",
        "query_count": 45,
        "last_queried": "2024-08-15T14:20:00Z"
      }
    ]
  },
  "performance": {
    "average_search_time": 0.12,
    "cache_hit_rate": 0.78,
    "index_efficiency": 0.92
  },
  "metadata": {
    "description": "Academic research papers collection",
    "tags": ["academic", "research", "ml"],
    "custom_fields": {}
  }
}
```

---

### DELETE `/collections/{collection_name}`
**Description:** Delete a collection and all its documents.

**Path Parameters:**
- `collection_name` (required): Name of the collection to delete

**Query Parameters:**
- `confirm` (required): Must be set to "true" for safety
- `remove_files` (optional): Also remove stored files (default: false)

**Response:**
```json
{
  "collection_name": "old_collection",
  "status": "deleted",
  "deleted_items": {
    "documents": 25,
    "chunks": 375,
    "embeddings": 375,
    "files": 25
  },
  "freed_space_mb": 87.3,
  "deletion_time": 2.34
}
```

---

## 🔌 WebSocket Endpoints

### WS `/ws/query`
**Description:** Real-time query processing with bidirectional communication.

**Connection Headers:**
- `Authorization: Bearer <token>` (if authentication enabled)
- `X-Collection-Name: <collection>` (optional)

**Message Types:**

#### Client → Server Messages

**Query Message:**
```json
{
  "type": "query",
  "request_id": "req_12345",
  "data": {
    "query": "What is machine learning?",
    "mode": "balanced",
    "stream": true,
    "max_results": 10
  }
}
```

**Configuration Message:**
```json
{
  "type": "config",
  "request_id": "req_12346",
  "data": {
    "collection_name": "research_papers",
    "language": "en",
    "response_format": "with_sources"
  }
}
```

**Control Message:**
```json
{
  "type": "control",
  "request_id": "req_12347",
  "data": {
    "action": "cancel" // or "pause", "resume"
  }
}
```

#### Server → Client Messages

**Status Updates:**
```json
{"type": "status", "request_id": "req_12345", "data": "Analyzing query..."}
{"type": "status", "request_id": "req_12345", "data": "Searching documents..."}
```

**Streaming Response:**
```json
{"type": "token", "request_id": "req_12345", "data": "Machine"}
{"type": "token", "request_id": "req_12345", "data": " learning"}
{"type": "citation", "request_id": "req_12345", "data": {"number": 1, "source": "ml_paper.pdf"}}
```

**Completion:**
```json
{
  "type": "complete",
  "request_id": "req_12345", 
  "data": {
    "total_time": 2.1,
    "tokens_generated": 156,
    "sources_used": 3
  }
}
```

**Error Handling:**
```json
{
  "type": "error",
  "request_id": "req_12345",
  "data": {
    "code": "PROCESSING_ERROR",
    "message": "Failed to generate response",
    "details": {...}
  }
}
```

---

### WS `/ws/upload`
**Description:** Real-time document upload with progress updates.

**Upload Progress Messages:**
```json
{"type": "upload_progress", "filename": "document.pdf", "progress": 0.45}
{"type": "processing_progress", "filename": "document.pdf", "stage": "text_extraction", "progress": 0.20}
{"type": "embedding_progress", "filename": "document.pdf", "chunks_processed": 8, "total_chunks": 15}
{"type": "upload_complete", "filename": "document.pdf", "document_id": "doc_051"}
```

---

## 🚨 Error Response Schema

All endpoints return errors in a consistent format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "specific_field",
      "value": "invalid_value",
      "constraint": "validation_rule"
    },
    "request_id": "req_12345",
    "timestamp": "2024-08-15T10:30:00Z",
    "path": "/api/v1/query",
    "method": "POST"
  }
}
```

**Common Error Codes:**
- `VALIDATION_ERROR`: Request validation failed
- `AUTHENTICATION_REQUIRED`: Authentication required
- `AUTHORIZATION_FAILED`: Insufficient permissions  
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `PROCESSING_ERROR`: Internal processing error
- `SERVICE_UNAVAILABLE`: Required service unavailable
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `TIMEOUT_ERROR`: Request timeout

---

**This endpoint specification provides complete details for integrating with the Modern RAG Application API.**
