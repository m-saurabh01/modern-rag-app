# EmbeddingService Documentation

## Overview
The `EmbeddingService` is the foundation of the modern RAG system, providing high-quality semantic embeddings using sentence-transformer models. It replaces the legacy TF-IDF approach with dense vector representations that capture semantic meaning and context.

## Class: EmbeddingService

### Purpose
Transform text into dense vector representations that enable semantic search, similarity matching, and context-aware retrieval. The service is optimized for CPU-only processing with 32GB RAM constraints.

## Architecture Decisions

### Why Sentence-Transformers?
- **Semantic Understanding**: Captures meaning beyond keyword matching
- **Pre-trained Models**: No training required, ready for production
- **CPU Optimization**: Efficient inference without GPU requirements
- **Industry Standard**: Used by major RAG implementations

### Memory Management Strategy
- **Lazy Loading**: Models loaded only when needed
- **Memory Monitoring**: Real-time usage tracking with configurable limits
- **Batch Processing**: Efficient processing of multiple texts
- **Resource Cleanup**: Explicit memory management and garbage collection

## Method Reference

### Constructor

```python
def __init__(self, model_name: Optional[str] = None)
```

**Purpose**: Initialize embedding service with specified model.

**Parameters**:
- `model_name` (Optional[str]): Sentence transformer model name. Defaults to `settings.model.embedding_model`

**Raises**:
- `ModelLoadingError`: If model fails to load
- `MemoryError`: If insufficient memory for model

**Usage Example**:
```python
# Use default model from settings
service = EmbeddingService()

# Use specific model
service = EmbeddingService("sentence-transformers/all-mpnet-base-v2")
```

**Architecture Notes**:
- Model loading is lazy (deferred until first use)
- Thread-safe initialization with locking mechanism
- Performance statistics tracking enabled by default

### Properties

#### `is_loaded`
```python
@property
def is_loaded(self) -> bool
```

**Purpose**: Check if model is loaded and ready for inference.

**Returns**: `True` if model loaded, `False` otherwise

**Performance Considerations**: O(1) operation, safe to call frequently

#### `model_info`
```python
@property
def model_info(self) -> Dict[str, Any]
```

**Purpose**: Get comprehensive model information and performance statistics.

**Returns**: Dictionary containing:
- `model_name`: Model identifier
- `is_loaded`: Current loading status
- `batch_size`: Configured batch size
- `embedding_dimension`: Vector dimension (if loaded)
- `max_sequence_length`: Maximum input length (if available)
- `device`: Computing device (CPU/GPU)
- `stats`: Performance statistics

**Usage Example**:
```python
service = EmbeddingService()
info = service.model_info

print(f"Model: {info['model_name']}")
print(f"Dimension: {info.get('embedding_dimension', 'Unknown')}")
print(f"Total embeddings generated: {info['stats']['total_embeddings']}")
```

### Core Methods

#### `load_model()`
```python
@log_performance("model_loading")
def load_model(self) -> None
```

**Purpose**: Explicitly load the sentence transformer model with memory monitoring.

**Raises**:
- `ModelLoadingError`: If model loading fails
- `MemoryError`: If insufficient memory

**Performance Characteristics**:
- **First call**: Downloads model if not cached (~90-420MB depending on model)
- **Subsequent calls**: Fast loading from disk cache
- **Memory usage**: Monitored during loading process
- **Warm-up**: Includes test inference for optimal performance

**Architecture Details**:
- Thread-safe with explicit locking
- Memory usage tracking before/after loading
- Model optimization for inference mode
- Automatic warm-up with test embedding

**Usage Example**:
```python
service = EmbeddingService()
service.load_model()  # Explicit loading

# Check if successful
if service.is_loaded:
    print("Model ready for inference")
```

#### `ensure_model_loaded()`
```python
async def ensure_model_loaded(self) -> None
```

**Purpose**: Asynchronously ensure model is loaded without blocking the event loop.

**Architecture Benefits**:
- Non-blocking model loading
- Uses thread pool executor
- Safe for async/await contexts
- Prevents event loop blocking during model initialization

**Usage Example**:
```python
async def process_documents():
    service = EmbeddingService()
    await service.ensure_model_loaded()  # Non-blocking
    # Model ready for use
```

#### `embed_texts()`
```python
@log_performance("text_embedding")
async def embed_texts(
    self, 
    texts: Union[str, List[str]],
    batch_size: Optional[int] = None,
    normalize_embeddings: bool = True,
    progress_callback: Optional[callable] = None
) -> Union[np.ndarray, List[np.ndarray]]
```

**Purpose**: Generate embeddings for text(s) with batch processing and progress tracking.

**Parameters**:
- `texts`: Single text string or list of texts to embed
- `batch_size`: Override default batch size for processing
- `normalize_embeddings`: Whether to normalize embeddings to unit length
- `progress_callback`: Optional callback function `(current: int, total: int)`

**Returns**:
- Single `np.ndarray` for string input
- `List[np.ndarray]` for list input

**Raises**:
- `EmbeddingError`: If embedding generation fails
- `MemoryError`: If memory limit exceeded during processing

**Performance Characteristics**:
- **Batch Processing**: Processes multiple texts efficiently
- **Memory Monitoring**: Checks memory usage before each batch
- **Progress Tracking**: Real-time progress callbacks
- **Async-Friendly**: Non-blocking execution in thread pool

**Usage Examples**:
```python
# Single text
service = EmbeddingService()
embedding = await service.embed_texts("Hello world")
# Returns: np.ndarray shape (384,)

# Multiple texts with progress tracking
texts = ["Text 1", "Text 2", "Text 3"]

def progress_handler(current, total):
    print(f"Progress: {current}/{total}")

embeddings = await service.embed_texts(
    texts, 
    batch_size=100,
    progress_callback=progress_handler
)
# Returns: List[np.ndarray] length 3

# Custom batch size for memory optimization
large_texts = ["..."] * 1000
embeddings = await service.embed_texts(
    large_texts,
    batch_size=50  # Smaller batches for memory constraint
)
```

**Architecture Notes**:
- Automatic model loading if not already loaded
- Batch processing prevents memory overflow
- Progress callbacks enable UI updates
- Memory monitoring prevents system crashes

#### `embed_documents()`
```python
async def embed_documents(
    self,
    documents: List[Dict[str, Any]],
    text_field: str = 'content',
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]
```

**Purpose**: Embed documents while preserving metadata and document structure.

**Parameters**:
- `documents`: List of document dictionaries
- `text_field`: Field name containing text to embed (default: 'content')
- `progress_callback`: Optional progress callback

**Returns**: Documents with added 'embedding' field

**Architecture Benefits**:
- Preserves document metadata
- Handles missing or empty text fields gracefully
- Maintains document structure
- Enables batch processing of document collections

**Usage Example**:
```python
documents = [
    {"content": "Document 1", "title": "Title 1", "author": "Author 1"},
    {"content": "Document 2", "title": "Title 2", "author": "Author 2"},
    {"description": "Alt text field", "title": "Title 3"}
]

# Default field ('content')
embedded_docs = await service.embed_documents(documents)

# Custom text field
alt_docs = await service.embed_documents(
    documents, 
    text_field="description"
)

# Result structure:
# {
#     "content": "Document 1",
#     "title": "Title 1", 
#     "author": "Author 1",
#     "embedding": [0.1, -0.2, 0.3, ...]  # 384-dimensional vector
# }
```

#### `get_embedding_dimension()`
```python
def get_embedding_dimension(self) -> int
```

**Purpose**: Get the dimension of embeddings produced by the model.

**Returns**: Integer dimension (e.g., 384 for MiniLM, 768 for MPNet)

**Raises**:
- `ModelLoadingError`: If model loading fails

**Architecture Notes**:
- Automatically loads model if not already loaded
- Cached after first call
- Essential for vector database configuration

**Usage Example**:
```python
service = EmbeddingService()
dim = service.get_embedding_dimension()  # 384
print(f"Embedding dimension: {dim}")

# Use for vector database setup
await vector_store.create_collection(dimension=dim)
```

### Utility Methods

#### `cleanup()`
```python
def cleanup(self) -> None
```

**Purpose**: Clean up model and free memory resources.

**Architecture Benefits**:
- Explicit memory management
- Prevents memory leaks
- Thread-safe cleanup
- Forces garbage collection

**Usage Example**:
```python
service = EmbeddingService()
# ... use service
service.cleanup()  # Free memory
```

## Performance Characteristics

### Memory Usage
| Model | RAM Usage | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | ~90MB | Fast | Good | Development, lightweight |
| `all-mpnet-base-v2` | ~420MB | Medium | Better | Production, high quality |
| `multi-qa-MiniLM-L6-cos-v1` | ~90MB | Fast | Good | Q&A specific |

### Processing Speed (32GB RAM, CPU-only)
- **Single text**: <10ms
- **Batch (100 texts)**: ~500ms
- **Large batch (1000 texts)**: ~5s
- **Model loading**: 2-5s (first time), <1s (cached)

### Batch Size Recommendations
```python
# Conservative (safe for all configurations)
batch_size = 50

# Balanced (good performance/memory trade-off)
batch_size = 100

# Aggressive (monitor memory usage)
batch_size = 200
```

## Error Handling

### Exception Hierarchy
```
EmbeddingError (base)
├── ModelLoadingError
│   ├── Model not found
│   ├── Download failure
│   └── Insufficient memory
└── Memory error during processing
```

### Common Error Scenarios

#### Model Loading Issues
```python
try:
    service = EmbeddingService("invalid-model-name")
    service.load_model()
except ModelLoadingError as e:
    print(f"Model loading failed: {e.message}")
    print(f"Error code: {e.error_code}")
    print(f"Model name: {e.details['model_name']}")
```

#### Memory Constraints
```python
try:
    large_texts = ["..."] * 10000
    embeddings = await service.embed_texts(large_texts)
except MemoryError as e:
    print(f"Memory limit exceeded: {e.message}")
    print(f"Current usage: {e.details['memory_usage_mb']}MB")
    # Reduce batch size or process in smaller chunks
```

## Best Practices

### 1. Service Lifecycle Management
```python
# Application startup
service = EmbeddingService()
await service.ensure_model_loaded()

# During operation
embeddings = await service.embed_texts(texts)

# Application shutdown
service.cleanup()
```

### 2. Memory Optimization
```python
# Monitor memory usage
info = service.model_info
current_memory = info['stats'].get('memory_usage', 0)

# Adjust batch size based on available memory
available_memory_gb = 32 - current_memory
batch_size = min(200, int(available_memory_gb * 10))

embeddings = await service.embed_texts(
    texts, 
    batch_size=batch_size
)
```

### 3. Progress Tracking for Long Operations
```python
def create_progress_handler(operation_name: str):
    def handler(current: int, total: int):
        percent = (current / total) * 100
        print(f"{operation_name}: {percent:.1f}% ({current}/{total})")
    return handler

# Use with embedding operations
progress_handler = create_progress_handler("Document Embedding")
embeddings = await service.embed_texts(
    documents,
    progress_callback=progress_handler
)
```

### 4. Error Recovery Strategies
```python
async def robust_embedding(service, texts, max_retries=3):
    """Robust embedding with retry logic and batch size reduction."""
    batch_size = 100
    
    for attempt in range(max_retries):
        try:
            return await service.embed_texts(texts, batch_size=batch_size)
        except MemoryError:
            batch_size = max(10, batch_size // 2)  # Reduce batch size
            print(f"Memory error, reducing batch size to {batch_size}")
        except EmbeddingError as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(1)  # Brief delay before retry
```

## Integration Patterns

### With Vector Databases
```python
# Initialize services
embedding_service = EmbeddingService()
vector_store = ChromaVectorStore()

# Create collection with correct dimension
dimension = embedding_service.get_embedding_dimension()
await vector_store.create_collection(dimension=dimension)

# Process and store documents
embedded_docs = await embedding_service.embed_documents(documents)
await vector_store.add_documents([
    VectorDocument(
        id=doc['id'],
        content=doc['content'], 
        embedding=doc['embedding'],
        metadata=doc
    ) for doc in embedded_docs
])
```

### With Search Services
```python
# Embed query for search
query = "What is machine learning?"
query_embedding = await embedding_service.embed_texts(query)

# Search vector database
results = await vector_store.search(
    query_embedding=query_embedding.tolist(),
    limit=10
)
```

This comprehensive documentation provides developers with everything needed to effectively use and integrate the EmbeddingService in production applications.
