# Settings Configuration Documentation

## Overview

The `Settings` class provides centralized configuration management for the Modern RAG application using Pydantic's `BaseSettings`. It implements environment-based configuration with validation, type checking, and comprehensive default values suitable for production deployment.

## Architecture

The settings system implements a hierarchical configuration approach:

- **Environment Variables**: Primary configuration source with prefix support
- **Default Values**: Fallback values for all configuration options
- **Type Validation**: Automatic type checking and conversion
- **Nested Configuration**: Grouped settings for different application components
- **Development/Production Profiles**: Environment-specific configuration presets

## Class Structure

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODERN_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
```

## Configuration Sections

### Application Settings

#### `app_name: str = "Modern RAG Application"`

**Purpose**: Defines the application name used in logging, API documentation, and system identification.

**What it achieves**:
- Provides consistent application identification across logs and interfaces
- Enables application branding and identification in multi-service environments
- Supports deployment tracking and service discovery

**Environment Variable**: `MODERN_RAG_APP_NAME`

**Usage Example**:
```bash
export MODERN_RAG_APP_NAME="Production RAG Service"
```

#### `app_version: str = "1.0.0"`

**Purpose**: Tracks application version for deployment management and compatibility checking.

**What it achieves**:
- Enables version tracking in logs and API responses
- Supports compatibility checking between components
- Facilitates deployment and rollback procedures
- Provides version information for monitoring and debugging

**Environment Variable**: `MODERN_RAG_APP_VERSION`

#### `debug: bool = False`

**Purpose**: Controls debug mode activation for development and troubleshooting.

**What it achieves**:
- Enables detailed logging and error tracing in development
- Controls debug-specific features and verbose output
- Affects performance optimizations and caching behavior
- Provides development-friendly error messages and stack traces

**Environment Variable**: `MODERN_RAG_DEBUG`

**Impact on System Behavior**:
- **Debug Mode ON**: Detailed logs, stack traces, development features enabled
- **Debug Mode OFF**: Production logging, optimized performance, secure error messages

### Embedding Service Configuration

#### `embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"`

**Purpose**: Specifies the sentence transformer model for text embedding generation.

**What it achieves**:
- Defines the semantic embedding model for text vectorization
- Controls embedding quality and dimensionality characteristics
- Enables model selection based on performance and accuracy requirements
- Supports offline operation with locally cached models

**Environment Variable**: `MODERN_RAG_EMBEDDING_MODEL`

**Model Selection Guidelines**:
- **all-MiniLM-L6-v2**: Fast, lightweight, good for general use (384 dimensions)
- **all-mpnet-base-v2**: Higher accuracy, larger model (768 dimensions)
- **distilbert-base-nli-stsb-mean-tokens**: Balanced performance (768 dimensions)

**Usage Example**:
```bash
export MODERN_RAG_EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
```

#### `embedding_dimension: int = 384`

**Purpose**: Defines the dimensionality of embedding vectors for consistency and storage optimization.

**What it achieves**:
- Ensures consistent vector dimensions across the application
- Optimizes storage requirements and indexing performance
- Validates compatibility between models and vector storage
- Supports vector dimension-specific optimizations

**Environment Variable**: `MODERN_RAG_EMBEDDING_DIMENSION`

**Dimension Considerations**:
- **384**: Faster processing, lower memory usage, good accuracy
- **768**: Higher accuracy, more memory usage, slower processing
- **1024+**: Maximum accuracy, significant resource requirements

#### `batch_size: int = 32`

**Purpose**: Controls batch processing size for embedding generation to optimize memory usage and throughput.

**What it achieves**:
- Balances memory consumption with processing efficiency
- Optimizes GPU/CPU utilization for embedding generation
- Prevents memory overflow with large document collections
- Enables fine-tuning for specific hardware configurations

**Environment Variable**: `MODERN_RAG_BATCH_SIZE`

**Batch Size Guidelines**:
- **32GB RAM, CPU-only**: 16-32 (default: 32)
- **16GB RAM**: 8-16
- **8GB RAM**: 4-8
- **GPU Available**: 64-128

### Vector Database Configuration

#### `vector_db_type: str = "chroma"`

**Purpose**: Selects the vector database backend for embedding storage and retrieval.

**What it achieves**:
- Enables switching between different vector database solutions
- Supports development and production database configurations
- Allows optimization for specific use cases and performance requirements
- Provides flexibility for scaling and deployment strategies

**Environment Variable**: `MODERN_RAG_VECTOR_DB_TYPE`

**Supported Database Types**:
- **chroma**: Development-friendly, lightweight, local storage
- **qdrant**: Production-ready, high-performance, distributed
- **pinecone**: Cloud-managed, scalable (requires API key)
- **weaviate**: Open-source, GraphQL, cloud or on-premise

#### `chroma_persist_directory: str = "data/chroma_db"`

**Purpose**: Specifies persistent storage location for ChromaDB vector database.

**What it achieves**:
- Enables persistent storage of embeddings across application restarts
- Provides data durability and recovery capabilities
- Supports backup and migration procedures
- Allows custom storage locations for deployment flexibility

**Environment Variable**: `MODERN_RAG_CHROMA_PERSIST_DIRECTORY`

**Storage Considerations**:
- **Local Development**: `./data/chroma_db`
- **Docker Deployment**: `/app/data/chroma_db`
- **Production**: `/var/lib/modern-rag/chroma_db`

#### `collection_name: str = "documents"`

**Purpose**: Defines the default collection name for document storage in vector database.

**What it achieves**:
- Organizes documents into logical collections
- Supports multi-tenant or multi-domain document organization
- Enables collection-specific configuration and management
- Provides namespace isolation for different document types

**Environment Variable**: `MODERN_RAG_COLLECTION_NAME`

### Text Processing Configuration

#### `chunk_size: int = 1000`

**Purpose**: Defines the default chunk size for text segmentation in characters.

**What it achieves**:
- Controls granularity of document segmentation for embedding
- Balances context preservation with processing efficiency
- Optimizes retrieval accuracy and relevance
- Enables tuning for specific document types and use cases

**Environment Variable**: `MODERN_RAG_CHUNK_SIZE`

**Chunk Size Guidelines**:
- **500-800**: Short documents, precise retrieval
- **1000-1500**: General purpose, balanced approach (default: 1000)
- **2000+**: Long-form content, context-heavy documents

#### `chunk_overlap: int = 200`

**Purpose**: Specifies overlap between consecutive chunks to preserve context boundaries.

**What it achieves**:
- Prevents context loss at chunk boundaries
- Improves retrieval quality by maintaining semantic continuity
- Reduces information fragmentation in document processing
- Enables better semantic understanding across chunk boundaries

**Environment Variable**: `MODERN_RAG_CHUNK_OVERLAP`

**Overlap Strategies**:
- **10-20% of chunk_size**: Standard approach (default: 200 for 1000 chunk size)
- **Sentence-boundary aware**: Overlap at natural language boundaries
- **Semantic overlap**: Context-aware boundary detection

### Logging Configuration

#### `log_level: str = "INFO"`

**Purpose**: Controls the verbosity level of application logging.

**What it achieves**:
- Manages logging verbosity for different deployment environments
- Balances information availability with log volume
- Enables debugging capabilities when needed
- Supports production logging requirements

**Environment Variable**: `MODERN_RAG_LOG_LEVEL`

**Log Level Hierarchy**:
- **DEBUG**: Detailed debugging information, development use
- **INFO**: General information, normal operation (default)
- **WARNING**: Warning conditions, potential issues
- **ERROR**: Error conditions, system problems
- **CRITICAL**: Critical errors, system failure

#### `log_format: str = "json"`

**Purpose**: Specifies the logging output format for structured log processing.

**What it achieves**:
- Enables structured log parsing and analysis
- Supports log aggregation and monitoring systems
- Provides machine-readable log format for automated processing
- Facilitates log indexing and searching in production systems

**Environment Variable**: `MODERN_RAG_LOG_FORMAT`

**Format Options**:
- **json**: Structured JSON format for production systems
- **text**: Human-readable format for development
- **console**: Color-coded console output for local development

## Configuration Loading and Validation

### Environment File Loading

**Purpose**: Loads configuration from `.env` files for development convenience.

**What it achieves**:
- Simplifies development environment setup
- Provides secure credential management for local development
- Enables version-controlled configuration templates
- Supports environment-specific configuration files

**File Loading Priority**:
1. Environment variables (highest priority)
2. `.env` file in application root
3. Default values (lowest priority)

**Example `.env` file**:
```bash
MODERN_RAG_APP_NAME="Development RAG Service"
MODERN_RAG_DEBUG=true
MODERN_RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
MODERN_RAG_LOG_LEVEL="DEBUG"
MODERN_RAG_CHUNK_SIZE=800
```

### Validation and Type Checking

**Purpose**: Ensures configuration values are valid and properly typed.

**What it achieves**:
- Prevents runtime errors from invalid configuration
- Provides clear error messages for configuration problems
- Ensures type safety across the application
- Validates configuration constraints and dependencies

**Validation Features**:
- **Type Conversion**: Automatic string-to-type conversion from environment variables
- **Value Validation**: Range and format validation for configuration values
- **Dependency Checking**: Validation of configuration interdependencies
- **Error Reporting**: Clear error messages for configuration problems

### Configuration Access Patterns

#### Singleton Pattern Usage

```python
# Application initialization
settings = Settings()

# Access throughout application
app_name = settings.app_name
embedding_model = settings.embedding_model
```

#### Dependency Injection

```python
# Service initialization with settings
embedding_service = EmbeddingService(
    model_name=settings.embedding_model,
    batch_size=settings.batch_size
)

vector_store = ChromaStore(
    persist_directory=settings.chroma_persist_directory,
    collection_name=settings.collection_name
)
```

## Environment-Specific Configuration

### Development Configuration

```bash
MODERN_RAG_APP_NAME="RAG Development"
MODERN_RAG_DEBUG=true
MODERN_RAG_LOG_LEVEL="DEBUG"
MODERN_RAG_LOG_FORMAT="console"
MODERN_RAG_VECTOR_DB_TYPE="chroma"
MODERN_RAG_BATCH_SIZE=16
```

### Production Configuration

```bash
MODERN_RAG_APP_NAME="Production RAG Service"
MODERN_RAG_DEBUG=false
MODERN_RAG_LOG_LEVEL="INFO"
MODERN_RAG_LOG_FORMAT="json"
MODERN_RAG_VECTOR_DB_TYPE="qdrant"
MODERN_RAG_BATCH_SIZE=32
MODERN_RAG_CHROMA_PERSIST_DIRECTORY="/var/lib/modern-rag/chroma_db"
```

### Docker Configuration

```bash
MODERN_RAG_APP_NAME="Containerized RAG Service"
MODERN_RAG_DEBUG=false
MODERN_RAG_LOG_LEVEL="INFO"
MODERN_RAG_LOG_FORMAT="json"
MODERN_RAG_CHROMA_PERSIST_DIRECTORY="/app/data/chroma_db"
```

## Configuration Best Practices

### Security Considerations
- **Sensitive Data**: Use environment variables for secrets and credentials
- **Default Values**: Provide secure defaults that work in isolated environments
- **Validation**: Implement strict validation for security-relevant configuration
- **Logging**: Avoid logging sensitive configuration values

### Performance Optimization
- **Resource Limits**: Configure based on available system resources
- **Batch Sizes**: Tune based on memory and processing capabilities
- **Model Selection**: Choose models appropriate for hardware constraints
- **Storage Paths**: Use fast storage for frequently accessed data

### Deployment Strategies
- **Environment Isolation**: Use environment-specific configuration sets
- **Configuration Management**: Version control configuration templates
- **Monitoring**: Include configuration in health checks and monitoring
- **Documentation**: Maintain comprehensive configuration documentation

### Troubleshooting Common Issues

#### Configuration Loading Problems
```python
# Debug configuration loading
try:
    settings = Settings()
    print(f"Loaded settings: {settings.dict()}")
except Exception as e:
    print(f"Configuration error: {e}")
```

#### Environment Variable Issues
- **Missing Variables**: Check environment variable names and prefixes
- **Type Conversion**: Verify environment variables are properly formatted
- **File Permissions**: Ensure `.env` files are readable
- **Path Resolution**: Verify file paths are absolute or properly relative

#### Model and Storage Configuration
- **Model Availability**: Verify embedding models are accessible offline
- **Storage Permissions**: Ensure write permissions for persist directories
- **Disk Space**: Monitor storage usage for vector databases
- **Memory Limits**: Configure batch sizes based on available RAM

This configuration system provides a robust, flexible foundation for the Modern RAG application with comprehensive validation, environment support, and production-ready defaults.
