# RAG Orchestrator Service

The RAG Orchestrator is the central coordination service that manages the complete Retrieval-Augmented Generation pipeline, integrating all phases from document processing to final response generation.

## Overview

The RAG Orchestrator (`services/rag_orchestrator.py`) serves as the main entry point for processing queries through the complete RAG pipeline. It coordinates multiple specialized services to deliver intelligent responses with optimal performance and accuracy.

## Key Responsibilities

### 🎯 **Pipeline Coordination**
- Orchestrates the complete RAG workflow from query to response
- Manages service dependencies and execution order
- Handles error recovery and fallback strategies
- Coordinates streaming and batch processing modes

### ⚡ **Performance Management**
- Implements multiple processing modes (Speed/Balanced/Comprehensive)
- Manages resource allocation and optimization
- Provides response time guarantees (<1s, <3s, <10s)
- Handles concurrent request processing

### 📊 **Response Generation**
- Integrates retrieval results with LLM generation
- Supports multiple response formats (text-only, with sources, with analysis)
- Manages streaming responses for real-time interaction
- Provides comprehensive response metadata

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  RAG Orchestrator                          │
├─────────────────────────────────────────────────────────────┤
│  Query Processing Pipeline                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Query     │──│ Intelligent │──│ Intelligent │        │
│  │  Analyzer   │  │ Retriever   │  │ Summarizer  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  Document Processing Pipeline                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    PDF      │──│ Text/Doc    │──│  Chunking   │        │
│  │ Processor   │  │ Analyzer    │  │  Service    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Processing Modes

### 🚀 **Speed Mode** (`<1s` response time)
- **Query Processing**: Basic intent classification
- **Retrieval**: Fast semantic search with limited results
- **Generation**: Concise responses with essential information
- **Use Case**: Quick queries, real-time interactions

### ⚖️ **Balanced Mode** (`<3s` response time)
- **Query Processing**: Full intent analysis and entity extraction
- **Retrieval**: Multi-modal search with relevance ranking
- **Generation**: Comprehensive responses with source attribution
- **Use Case**: Standard queries, balanced speed/accuracy

### 🔍 **Comprehensive Mode** (`<10s` response time)
- **Query Processing**: Advanced analysis with query enhancement
- **Retrieval**: Deep search with cross-document relationships
- **Generation**: Detailed responses with full analysis
- **Use Case**: Complex queries, research scenarios

## Key Methods

### `async def process_query()`
Main query processing pipeline with full orchestration:

```python
async def process_query(
    self,
    query: str,
    mode: PipelineMode = PipelineMode.BALANCED,
    response_format: ResponseFormat = ResponseFormat.WITH_SOURCES,
    include_metadata: bool = True
) -> RAGResponse
```

**Features:**
- Complete pipeline orchestration
- Performance mode selection
- Comprehensive error handling
- Response format customization

### `async def process_documents()`
Document ingestion and processing pipeline:

```python
async def process_documents(
    self,
    documents: List[Path],
    batch_size: int = 10,
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
) -> ProcessingResult
```

**Features:**
- Batch document processing
- Progress monitoring and reporting
- Error recovery for failed documents
- Metadata extraction and storage

### `async def stream_response()`
Streaming response generation for real-time interaction:

```python
async def stream_response(
    self,
    query: str,
    mode: PipelineMode = PipelineMode.BALANCED
) -> AsyncGenerator[StreamChunk, None]
```

**Features:**
- Real-time response streaming
- Progressive result delivery
- Error handling during streaming
- Cancellation support

## Response Formats

### 📝 **Text Only**
Simple text response for basic use cases:
```python
{
    "response": "Generated answer text",
    "processing_time": 0.8,
    "mode": "balanced"
}
```

### 📚 **With Sources**
Response including source attribution:
```python
{
    "response": "Generated answer with citations",
    "sources": [
        {
            "content": "Source text chunk",
            "metadata": {"source": "document.pdf", "page": 1},
            "score": 0.95
        }
    ],
    "processing_time": 1.2,
    "confidence": 0.87
}
```

### 🔍 **With Analysis**
Comprehensive response with full analysis:
```python
{
    "response": "Detailed generated answer",
    "sources": [...],
    "query_analysis": {
        "intent": "factual",
        "entities": ["entity1", "entity2"],
        "enhancement": "expanded query terms"
    },
    "retrieval_analysis": {
        "strategy_used": "multi_modal",
        "total_results": 15,
        "ranking_factors": ["semantic", "structural"]
    },
    "processing_metadata": {...}
}
```

## Integration Points

### 🔗 **Service Dependencies**
- **Query Analyzer**: Intent classification and enhancement
- **Intelligent Retriever**: Multi-modal search and ranking
- **Intelligent Summarizer**: LLM-powered response generation
- **Document Services**: PDF processing and analysis
- **Vector Store**: Document storage and retrieval

### 📡 **API Integration**
- **FastAPI Endpoints**: Direct integration with REST API
- **WebSocket Support**: Real-time streaming responses
- **Batch Processing**: Background document processing
- **Health Monitoring**: Service status and performance metrics

## Performance Characteristics

| Mode | Response Time | Accuracy | Memory Usage | CPU Usage |
|------|---------------|----------|--------------|-----------|
| Speed | <1s | 75-80% | Low | Low |
| Balanced | <3s | 85-90% | Moderate | Moderate |
| Comprehensive | <10s | 95%+ | High | High |

## Error Handling

### 🛡️ **Graceful Degradation**
- Service failures trigger fallback strategies
- Performance mode downgrade on resource constraints
- Partial results delivery when possible
- Clear error reporting with context

### 🔄 **Recovery Strategies**
- Automatic retry with exponential backoff
- Service health monitoring and recovery
- Alternative processing paths
- Resource cleanup and management

## Configuration

### Environment Variables
```bash
# Performance tuning
RAG_DEFAULT_MODE=balanced
RAG_MAX_CONCURRENT_REQUESTS=50
RAG_TIMEOUT_SECONDS=30

# Response configuration
RAG_DEFAULT_RESPONSE_FORMAT=with_sources
RAG_INCLUDE_METADATA=true

# Resource limits
RAG_MAX_MEMORY_MB=8192
RAG_MAX_CPU_PERCENT=80
```

### Service Configuration
```python
@dataclass
class OrchestratorConfig:
    default_mode: PipelineMode = PipelineMode.BALANCED
    max_concurrent_requests: int = 50
    timeout_seconds: int = 30
    enable_streaming: bool = True
    include_performance_metrics: bool = True
```

## Usage Examples

### Basic Query Processing
```python
orchestrator = RAGOrchestrator()

# Simple query
response = await orchestrator.process_query(
    query="What is the main topic of the document?",
    mode=PipelineMode.BALANCED
)

print(response.response)
print(f"Sources: {len(response.sources)}")
```

### Streaming Response
```python
async def stream_example():
    async for chunk in orchestrator.stream_response(
        query="Explain the key findings",
        mode=PipelineMode.COMPREHENSIVE
    ):
        if chunk.type == "text":
            print(chunk.content, end="", flush=True)
        elif chunk.type == "metadata":
            print(f"\n[Processing: {chunk.status}]")
```

### Document Processing
```python
# Process multiple documents
documents = [Path("doc1.pdf"), Path("doc2.pdf")]
result = await orchestrator.process_documents(
    documents=documents,
    batch_size=5,
    processing_mode=ProcessingMode.COMPREHENSIVE
)

print(f"Processed {result.successful_count} documents")
print(f"Failed: {result.failed_count}")
```

## Monitoring & Observability

### Performance Metrics
- Query processing times by mode
- Service response times and success rates
- Resource usage (memory, CPU, I/O)
- Concurrent request handling

### Business Metrics
- Query success and failure rates
- Response quality scores
- User satisfaction metrics
- Document processing throughput

## Related Documentation
- [Query Analyzer](query_analyzer.md) - Intent classification and enhancement
- [Intelligent Retriever](intelligent_retriever.md) - Multi-modal search capabilities
- [Intelligent Summarizer](intelligent_summarizer.md) - LLM-powered response generation
- [API Endpoints](../api/endpoints.md) - REST API integration
- [Configuration Guide](../configuration.md) - Environment setup

---

**The RAG Orchestrator is the heart of the Modern RAG Application, providing seamless integration between all system components while maintaining high performance and reliability.**
