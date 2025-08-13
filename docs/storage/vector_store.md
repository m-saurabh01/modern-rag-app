# Vector Store Documentation

## Overview

The Modern RAG application implements a flexible vector storage system with support for multiple vector database backends. The system provides a common interface for vector operations while allowing easy switching between different storage solutions based on requirements.

## Architecture

### Abstract Base Class: VectorStore

The `VectorStore` abstract base class defines the common interface for all vector database implementations:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

class VectorStore(ABC):
    """Abstract base class for vector database implementations."""
```

#### Core Methods
- `add_documents()`: Store documents with their embeddings
- `similarity_search()`: Find similar documents by vector similarity
- `similarity_search_with_score()`: Search with similarity scores
- `delete_documents()`: Remove documents from the store
- `get_collection_stats()`: Retrieve collection statistics
- `health_check()`: Verify store connectivity and status

### ChromaDB Implementation

The `ChromaVectorStore` class provides a complete ChromaDB implementation with persistent storage and advanced features.

#### Key Features
- **Persistent Storage**: Automatic persistence to disk
- **Metadata Filtering**: Advanced filtering capabilities
- **Batch Operations**: Efficient bulk operations
- **Memory Management**: Optimized for large document collections
- **Health Monitoring**: Comprehensive health checks

## Usage Examples

### Basic Operations

#### Initialization
```python
from storage.chroma_store import ChromaVectorStore

# Create vector store instance
vector_store = ChromaVectorStore(
    collection_name="documents",
    persist_directory="./vector_db"
)

# Initialize the store
await vector_store.initialize()
```

#### Adding Documents
```python
from models import Document

documents = [
    Document(
        content="AI is transforming healthcare...",
        metadata={"source": "healthcare.pdf", "page": 1}
    ),
    Document(
        content="Machine learning applications...",
        metadata={"source": "ml_guide.pdf", "page": 15}
    )
]

# Add documents with automatic embedding generation
await vector_store.add_documents(documents)
```

#### Similarity Search
```python
# Basic similarity search
results = await vector_store.similarity_search(
    query="healthcare AI applications",
    k=5
)

# Search with similarity scores
results_with_scores = await vector_store.similarity_search_with_score(
    query="healthcare AI applications",
    k=5,
    score_threshold=0.7
)
```

#### Metadata Filtering
```python
# Search with metadata filters
results = await vector_store.similarity_search(
    query="AI applications",
    k=10,
    filter={"source": "healthcare.pdf"}
)

# Complex filtering
results = await vector_store.similarity_search(
    query="machine learning",
    k=5,
    filter={
        "source": {"$in": ["ml_guide.pdf", "ai_handbook.pdf"]},
        "page": {"$gte": 10}
    }
)
```

### Advanced Operations

#### Batch Processing
```python
# Process large document collections in batches
large_document_set = load_large_document_collection()

await vector_store.add_documents(
    large_document_set,
    batch_size=100,
    show_progress=True
)
```

#### Collection Management
```python
# Get collection statistics
stats = await vector_store.get_collection_stats()
print(f"Total documents: {stats['count']}")
print(f"Embedding dimensions: {stats['dimension']}")

# Health check
health = await vector_store.health_check()
if health['status'] == 'healthy':
    print("Vector store is operational")
```

#### Document Updates and Deletion
```python
# Update documents
await vector_store.update_documents(
    document_ids=["doc_1", "doc_2"],
    documents=updated_documents
)

# Delete specific documents
await vector_store.delete_documents(["doc_1", "doc_3"])

# Clear entire collection
await vector_store.clear_collection()
```

## Configuration

### ChromaDB Configuration

#### Basic Configuration
```python
config = {
    "collection_name": "my_documents",
    "persist_directory": "./chroma_db",
    "embedding_function": "sentence-transformers",
    "distance_metric": "cosine"
}

vector_store = ChromaVectorStore(**config)
```

#### Advanced Configuration
```python
config = {
    "collection_name": "advanced_collection",
    "persist_directory": "/data/vector_db",
    "embedding_function": "all-MiniLM-L6-v2",
    "distance_metric": "cosine",
    "batch_size": 100,
    "max_retries": 3,
    "timeout_seconds": 30,
    "enable_compression": True,
    "auto_optimize": True
}
```

### Performance Tuning

#### Memory Optimization
```python
# Configure for large collections
vector_store = ChromaVectorStore(
    collection_name="large_collection",
    persist_directory="/data/vectors",
    batch_size=50,  # Smaller batches for memory efficiency
    enable_compression=True,
    memory_limit_mb=4000  # 4GB limit
)
```

#### Search Optimization
```python
# Optimize for search performance
search_config = {
    "enable_caching": True,
    "cache_size": 1000,
    "parallel_search": True,
    "max_parallel_queries": 4
}

results = await vector_store.similarity_search(
    query="search term",
    k=10,
    **search_config
)
```

## Error Handling

The vector store system uses comprehensive exception handling with specific error types:

### VectorStoreError
Base exception for vector database operations.

### VectorSearchError
Specific to search operations with query and collection context.

### VectorIndexError
For indexing and storage operations with operation details.

### Example Error Handling
```python
from core.exceptions import VectorStoreError, VectorSearchError

try:
    results = await vector_store.similarity_search("query")
except VectorSearchError as e:
    logger.error(f"Search failed: {e.message}")
    logger.error(f"Query: {e.details.get('query')}")
    logger.error(f"Collection: {e.details.get('collection_name')}")
except VectorStoreError as e:
    logger.error(f"Vector store error: {e.message}")
    logger.error(f"Store type: {e.details.get('vector_store_type')}")
```

## Integration Points

### With Embedding Service
```python
# Automatic embedding integration
from services.embedding_service import EmbeddingService

embedding_service = EmbeddingService()
vector_store = ChromaVectorStore(embedding_service=embedding_service)

# Documents are automatically embedded during storage
await vector_store.add_documents(documents)
```

### With Chunking Service
```python
# Pipeline integration
chunks = await chunking_service.chunk_documents(documents)
await vector_store.add_documents(chunks)
```

### With Search API
```python
# API endpoint integration
@router.post("/search")
async def search_documents(query: str, k: int = 5):
    try:
        results = await vector_store.similarity_search(query, k=k)
        return {"results": results}
    except VectorSearchError as e:
        raise HTTPException(status_code=500, detail=e.to_dict())
```

## Performance Monitoring

### Metrics Collection
```python
# Built-in metrics tracking
stats = await vector_store.get_performance_metrics()
print(f"Average search time: {stats['avg_search_time_ms']}ms")
print(f"Index size: {stats['index_size_mb']}MB")
print(f"Cache hit rate: {stats['cache_hit_rate']}%")
```

### Health Monitoring
```python
# Regular health checks
health = await vector_store.health_check()
if health['status'] != 'healthy':
    logger.warning(f"Vector store issues: {health['issues']}")
    # Implement alerting or recovery logic
```

## Migration and Backup

### Data Export
```python
# Export collection for backup
backup_data = await vector_store.export_collection(
    format="parquet",
    include_embeddings=True
)
```

### Data Import
```python
# Restore from backup
await vector_store.import_collection(
    backup_data,
    collection_name="restored_collection"
)
```

### Cross-Store Migration
```python
# Migrate between vector stores
source_store = ChromaVectorStore("source_collection")
target_store = QdrantVectorStore("target_collection")

documents = await source_store.get_all_documents()
await target_store.add_documents(documents)
```

## Testing

### Unit Tests
Located in `tests/test_storage/test_vector_store.py`

**Test Coverage:**
- CRUD operations
- Search functionality
- Error handling
- Performance benchmarks
- Health checks

### Integration Tests
- End-to-end document storage and retrieval
- Integration with embedding services
- Batch processing validation
- Concurrent access testing

## Best Practices

### Collection Design
1. **Naming**: Use descriptive collection names with versioning
2. **Metadata**: Design rich metadata schemas for effective filtering
3. **Partitioning**: Consider collection partitioning for large datasets
4. **Indexing**: Optimize metadata fields for common query patterns

### Performance Optimization
1. **Batch Size**: Tune batch sizes based on available memory
2. **Embedding Dimensions**: Choose appropriate embedding models
3. **Search Parameters**: Optimize k and threshold values
4. **Caching**: Enable caching for frequently accessed data

### Monitoring and Maintenance
1. **Regular Health Checks**: Implement automated health monitoring
2. **Performance Tracking**: Monitor search times and resource usage
3. **Data Validation**: Validate data integrity regularly
4. **Backup Strategy**: Implement regular backup procedures

## Troubleshooting

### Common Issues

1. **Slow Search Performance**
   - Check index optimization settings
   - Review batch size configuration
   - Enable query caching
   - Consider embedding dimension reduction

2. **Memory Issues**
   - Reduce batch sizes
   - Enable compression
   - Implement collection partitioning
   - Monitor memory usage patterns

3. **Connection Failures**
   - Verify database connectivity
   - Check timeout configurations
   - Implement retry logic
   - Monitor database health

4. **Inconsistent Results**
   - Validate embedding consistency
   - Check metadata indexing
   - Review distance metric settings
   - Verify data integrity

### Debugging Tools

```python
# Enable debug logging
import logging
logging.getLogger("vector_store").setLevel(logging.DEBUG)

# Detailed error information
try:
    results = await vector_store.similarity_search("query")
except Exception as e:
    logger.error(f"Search failed: {e}", exc_info=True)
    
# Performance profiling
with vector_store.profile_operation("search"):
    results = await vector_store.similarity_search("query")
```
