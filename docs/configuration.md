# Configuration Guide

## Overview
The Modern RAG application uses environment-based configuration with Pydantic settings validation. All configuration is centralized in the `config/settings.py` module.

## Configuration Structure

### Setting Categories
- **Application Settings**: Basic app configuration
- **Model Settings**: Embedding and LLM configuration  
- **Processing Settings**: Document processing parameters
- **Vector Database Settings**: ChromaDB/Qdrant configuration
- **Storage Settings**: File system and storage paths
- **Search Settings**: Search and retrieval parameters
- **Logging Settings**: Logging and monitoring configuration

## Environment Variables

### Quick Start
1. Copy `.env.example` to `.env`
2. Modify values for your environment
3. Application will automatically load configuration

### Required Variables
```bash
# Minimum required configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_BASE_URL=http://localhost:11434
VECTOR_DB_TYPE=chromadb
```

### Complete Configuration Reference

#### Application Settings
```bash
APP_NAME=Modern RAG Application
APP_VERSION=1.0.0
DEBUG=false
HOST=0.0.0.0
PORT=8000
```

#### Model Configuration
```bash
# Embedding Model (CPU-optimized options)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Fast, 384 dimensions
# EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2  # Better quality, 768 dimensions

EMBEDDING_BATCH_SIZE=100  # Batch size for embedding generation
OLLAMA_MODEL=llama3       # LLM model name
LLM_TEMPERATURE=0.7       # Generation creativity (0.0-1.0)
```

#### Processing Configuration
```bash
# Memory management for 32GB RAM
MAX_MEMORY_GB=24                    # Leave headroom for OS
PROCESSING_BATCH_SIZE=50            # Documents per batch
MAX_CONCURRENT_DOCUMENTS=10         # Parallel processing limit

# Text chunking
CHUNK_SIZE=1000                     # Target chunk size in tokens
CHUNK_OVERLAP=200                   # Overlap between chunks
```

#### Vector Database Configuration
```bash
# ChromaDB (recommended for development)
VECTOR_DB_TYPE=chromadb
CHROMADB_PERSIST_DIRECTORY=./storage/chromadb

# Qdrant (alternative for production)
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Configuration Validation

### Automatic Validation
The application validates all configuration on startup:
- **Type checking**: Ensures correct data types
- **Range validation**: Validates numeric ranges
- **Dependency checks**: Ensures related settings are consistent
- **Path validation**: Creates required directories

### Validation Examples
```python
# This will fail validation
CHUNK_OVERLAP=1200  # Cannot be >= CHUNK_SIZE (1000)

# This will pass
CHUNK_SIZE=1000
CHUNK_OVERLAP=200   # Valid overlap
```

### Error Handling
Invalid configuration results in:
1. **Detailed error messages** explaining the issue
2. **Suggested corrections** for common problems
3. **Application startup failure** (fail-fast principle)

## Environment-Specific Configuration

### Development Environment
```bash
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
ENABLE_METRICS=false
```

### Production Environment  
```bash
DEBUG=false
LOG_LEVEL=INFO
WORKERS=4
ENABLE_METRICS=true
LOG_FORMAT=json
```

### Testing Environment
```bash
VECTOR_DB_TYPE=chromadb
CHROMADB_PERSIST_DIRECTORY=./tests/test_storage
PROCESSING_BATCH_SIZE=5  # Smaller batches for testing
```

## Advanced Configuration

### Model Selection Guide

#### Embedding Models (CPU-Optimized)
```bash
# Fast & Lightweight (384 dimensions)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# Memory: ~90MB, Speed: Fast, Quality: Good

# Balanced (768 dimensions)  
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
# Memory: ~420MB, Speed: Medium, Quality: Better

# Question-Answering Optimized (384 dimensions)
EMBEDDING_MODEL=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
# Memory: ~90MB, Speed: Fast, Quality: Good for Q&A
```

### Performance Tuning

#### Memory Optimization
```bash
# Conservative settings for 32GB RAM
MAX_MEMORY_GB=20
PROCESSING_BATCH_SIZE=25
EMBEDDING_BATCH_SIZE=50

# Aggressive settings (monitor memory usage)
MAX_MEMORY_GB=28
PROCESSING_BATCH_SIZE=100
EMBEDDING_BATCH_SIZE=200
```

#### Processing Optimization
```bash
# CPU-bound tasks (document processing)
MAX_CONCURRENT_DOCUMENTS=8  # Slightly less than CPU cores

# I/O-bound tasks (embedding generation)
EMBEDDING_BATCH_SIZE=200    # Larger batches for efficiency
```

## Troubleshooting

### Common Configuration Issues

#### 1. Model Loading Errors
```bash
# Problem: Model not found or download fails
Error: "Model 'custom-model' not found"

# Solution: Use supported models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### 2. Memory Issues
```bash
# Problem: Out of memory during processing
Error: "Memory limit exceeded"

# Solution: Reduce batch sizes
MAX_MEMORY_GB=20
PROCESSING_BATCH_SIZE=25
EMBEDDING_BATCH_SIZE=50
```

#### 3. Vector Database Connection
```bash
# Problem: ChromaDB connection fails
Error: "Failed to connect to ChromaDB"

# Solution: Check persist directory permissions
CHROMADB_PERSIST_DIRECTORY=./storage/chromadb
# Ensure directory exists and is writable
```

### Configuration Testing
```python
# Test configuration loading
from config.settings import settings

# Verify all settings loaded correctly
print(f"Embedding model: {settings.model.embedding_model}")
print(f"Vector DB type: {settings.vector_db.vector_db_type}")
print(f"Memory limit: {settings.processing.max_memory_gb}GB")
```

## Best Practices

### Security
- **Never commit `.env` files** to version control
- **Use different secrets** for each environment
- **Rotate credentials regularly** in production

### Performance
- **Monitor memory usage** and adjust limits accordingly
- **Test different embedding models** for your use case
- **Profile processing batches** to find optimal sizes

### Maintainability  
- **Document custom settings** in comments
- **Use consistent naming** for environment variables
- **Group related settings** logically
