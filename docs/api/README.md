# API Reference Documentation

Complete reference for the Modern RAG Application REST API endpoints, WebSocket connections, and request/response schemas.

## 📚 Overview

The Modern RAG Application provides a comprehensive REST API built with FastAPI, featuring automatic OpenAPI documentation, request validation, and error handling. The API supports both synchronous and streaming responses for optimal user experience.

## 🔗 Base URL

```
http://localhost:8000  # Development
https://your-domain.com  # Production
```

## 📖 Interactive Documentation

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

## 🔑 Authentication

The API supports optional JWT-based authentication. When enabled, include the authorization header:

```bash
Authorization: Bearer <jwt_token>
```

## 📋 Endpoint Categories

### 🏥 **Health & Status**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/status` | GET | Detailed system status |
| `/metrics` | GET | Performance metrics |

### 📄 **Document Management**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/documents/upload` | POST | Upload and process documents |
| `/documents/{id}` | GET | Get document information |
| `/documents/{id}/chunks` | GET | Get document chunks |
| `/documents` | GET | List all documents |
| `/documents/{id}` | DELETE | Delete document |

### 🔍 **Query & Search**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Process query and generate response |
| `/query/stream` | POST | Streaming query processing |
| `/search` | POST | Search documents without LLM generation |
| `/analyze` | POST | Query analysis without retrieval |

### ⚙️ **Configuration**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/config` | GET | Get current configuration |
| `/config` | PUT | Update configuration |
| `/collections` | GET | List vector database collections |
| `/collections/{name}/info` | GET | Get collection information |

## 📝 Request/Response Schemas

### Query Processing

#### POST `/query`
Process a query through the complete RAG pipeline.

**Request Body:**
```json
{
  "query": "What is the main topic of the research paper?",
  "mode": "balanced",
  "response_format": "with_sources",
  "max_results": 10,
  "include_analysis": true,
  "stream": false
}
```

**Response:**
```json
{
  "response": "The main topic of the research paper is neural network transfer learning...",
  "sources": [
    {
      "content": "Transfer learning in neural networks...",
      "metadata": {
        "source": "research_paper.pdf",
        "page": 1,
        "chunk_id": "chunk_001"
      },
      "score": 0.95
    }
  ],
  "query_analysis": {
    "intent": "factual",
    "entities": ["neural networks", "transfer learning"],
    "confidence": 0.87
  },
  "processing_time": 1.23,
  "mode": "balanced"
}
```

#### POST `/query/stream`
Process a query with streaming response generation.

**Request Body:** Same as `/query` with `"stream": true`

**Response:** Server-Sent Events (SSE) stream
```
data: {"type": "analysis", "content": "Query analyzed: factual intent"}

data: {"type": "retrieval", "content": "Found 5 relevant chunks"}

data: {"type": "token", "content": "The"}

data: {"type": "token", "content": " main"}

data: {"type": "token", "content": " topic"}

data: {"type": "citation", "content": "[1] research_paper.pdf, page 1"}

data: {"type": "complete", "metadata": {"total_time": 2.1}}
```

### Document Management

#### POST `/documents/upload`
Upload and process documents for the vector database.

**Request:** Multipart form data
```
Content-Type: multipart/form-data

files: [file1.pdf, file2.pdf]
collection_name: "research_documents"
processing_mode: "balanced"
extract_metadata: true
```

**Response:**
```json
{
  "uploaded_documents": [
    {
      "filename": "file1.pdf",
      "document_id": "doc_001",
      "status": "processed",
      "chunks_created": 15,
      "metadata": {
        "pages": 10,
        "file_size": 2048576,
        "processing_time": 3.45
      }
    }
  ],
  "total_uploaded": 1,
  "total_failed": 0,
  "processing_time": 3.45
}
```

#### GET `/documents/{id}`
Retrieve document information and metadata.

**Response:**
```json
{
  "document_id": "doc_001",
  "filename": "research_paper.pdf",
  "upload_date": "2024-08-15T10:30:00Z",
  "status": "processed",
  "metadata": {
    "pages": 10,
    "file_size": 2048576,
    "content_type": "application/pdf",
    "language": "en",
    "author": "John Doe"
  },
  "processing_stats": {
    "chunks_created": 15,
    "processing_time": 3.45,
    "embeddings_generated": 15
  }
}
```

### Search Operations

#### POST `/search`
Search documents without LLM response generation.

**Request Body:**
```json
{
  "query": "neural networks",
  "collection_name": "research_documents",
  "limit": 10,
  "score_threshold": 0.7,
  "filters": {
    "source": ["research_paper.pdf"]
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "content": "Neural networks are computational models...",
      "metadata": {
        "source": "research_paper.pdf",
        "page": 1,
        "chunk_id": "chunk_001"
      },
      "score": 0.95,
      "distance": 0.05
    }
  ],
  "total_results": 5,
  "search_time": 0.12,
  "collection_info": {
    "name": "research_documents",
    "total_documents": 100
  }
}
```

## 🚨 Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "query",
      "issue": "Query cannot be empty"
    },
    "request_id": "req_12345",
    "timestamp": "2024-08-15T10:30:00Z"
  }
}
```

### HTTP Status Codes
- **200 OK**: Successful request
- **201 Created**: Resource created successfully
- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Validation error
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: Service temporarily unavailable

### Common Error Codes
- `VALIDATION_ERROR`: Request validation failed
- `DOCUMENT_NOT_FOUND`: Requested document doesn't exist
- `PROCESSING_ERROR`: Document processing failed
- `QUERY_ERROR`: Query processing failed
- `SERVICE_UNAVAILABLE`: Required service is unavailable
- `RATE_LIMIT_EXCEEDED`: Too many requests

## 🔄 WebSocket Connections

### `/ws/query`
Real-time query processing with bidirectional communication.

**Connection:** 
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/query');
```

**Message Format:**
```json
{
  "type": "query",
  "data": {
    "query": "What is transfer learning?",
    "mode": "balanced"
  },
  "request_id": "req_12345"
}
```

**Response Messages:**
```json
{"type": "status", "data": "Processing query...", "request_id": "req_12345"}
{"type": "token", "data": "Transfer", "request_id": "req_12345"}
{"type": "citation", "data": "[1] source.pdf", "request_id": "req_12345"}
{"type": "complete", "data": {"total_time": 2.1}, "request_id": "req_12345"}
```

## 🔧 Configuration Endpoints

### GET `/config`
Retrieve current system configuration.

**Response:**
```json
{
  "vector_db": {
    "type": "chromadb",
    "collection_name": "default_collection"
  },
  "embedding": {
    "model": "all-MiniLM-L6-v2",
    "dimension": 384
  },
  "llm": {
    "provider": "ollama",
    "model": "llama3",
    "base_url": "http://localhost:11434"
  },
  "performance": {
    "default_mode": "balanced",
    "max_concurrent_requests": 50
  }
}
```

## 📊 Metrics & Monitoring

### GET `/metrics`
Retrieve system performance metrics.

**Response:**
```json
{
  "system": {
    "uptime": 3600,
    "memory_usage": 4096,
    "cpu_usage": 25.5
  },
  "requests": {
    "total": 1500,
    "successful": 1485,
    "failed": 15,
    "average_response_time": 0.85
  },
  "processing": {
    "documents_processed": 50,
    "queries_processed": 1500,
    "embeddings_generated": 750
  },
  "database": {
    "total_documents": 50,
    "total_chunks": 750,
    "collection_size": 287.5
  }
}
```

## 🔒 Security Considerations

### Input Validation
- All requests are validated using Pydantic models
- File uploads are scanned and validated
- Query length and complexity limits enforced
- SQL injection and XSS protection

### Rate Limiting
- Per-IP request limits
- API key-based rate limiting
- Adaptive rate limiting based on resource usage
- Graceful degradation under load

### CORS Configuration
```json
{
  "allow_origins": ["http://localhost:3000"],
  "allow_methods": ["GET", "POST", "PUT", "DELETE"],
  "allow_headers": ["*"],
  "allow_credentials": true
}
```

## 📚 Client Examples

### Python Client
```python
import httpx
import asyncio

async def query_rag(query: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/query",
            json={
                "query": query,
                "mode": "balanced",
                "response_format": "with_sources"
            }
        )
        return response.json()

# Usage
result = asyncio.run(query_rag("What is machine learning?"))
print(result["response"])
```

### JavaScript Client
```javascript
// REST API
async function queryRAG(query) {
  const response = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: query,
      mode: 'balanced',
      response_format: 'with_sources'
    })
  });
  
  return await response.json();
}

// Streaming with Server-Sent Events
function streamQuery(query) {
  const eventSource = new EventSource(
    `http://localhost:8000/query/stream`,
    {
      method: 'POST',
      body: JSON.stringify({query, stream: true})
    }
  );
  
  eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    handleStreamChunk(data);
  };
}
```

### cURL Examples
```bash
# Basic query
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the research about?",
       "mode": "balanced"
     }'

# Upload document
curl -X POST "http://localhost:8000/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@document.pdf" \
     -F "collection_name=my_documents"

# Search without LLM
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "machine learning",
       "limit": 5,
       "score_threshold": 0.8
     }'
```

## Related Documentation
- [Endpoints Details](endpoints.md) - Detailed endpoint specifications
- [Authentication](authentication.md) - Authentication and authorization
- [Rate Limiting](rate_limiting.md) - API rate limiting and quotas
- [WebSocket Guide](websockets.md) - Real-time communication
- [Error Handling](../core/exception_handling.md) - Error handling strategies

---

**The Modern RAG API provides a comprehensive, well-documented interface for building intelligent document processing and retrieval applications.**
