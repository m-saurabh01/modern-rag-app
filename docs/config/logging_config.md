# Logging Configuration Documentation

## Overview

The `setup_logging` function provides centralized logging configuration for the Modern RAG application. It implements structured logging with JSON formatting, performance monitoring, and environment-aware configuration to support both development and production deployments.

## Architecture

The logging system implements a multi-layered approach:

- **Structured Logging**: JSON-formatted logs for machine processing
- **Performance Monitoring**: Request timing and resource usage tracking  
- **Environment Awareness**: Different log levels and formats based on environment
- **Centralized Configuration**: Single point of logging setup across the application
- **Production Ready**: Optimized for log aggregation and monitoring systems

## Function Structure

```python
def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    enable_performance_logging: bool = True
) -> None
```

## Function Documentation

### `setup_logging(level: str = "INFO", format_type: str = "json", enable_performance_logging: bool = True) -> None`

**Purpose**: Configures application-wide logging with structured output, performance monitoring, and environment-appropriate settings.

**Parameters**:
- `level` (str): Logging level - 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' (default: 'INFO')
- `format_type` (str): Log output format - 'json', 'text', 'console' (default: 'json')
- `enable_performance_logging` (bool): Enable performance and timing logs (default: True)

**What it achieves**:
- Establishes consistent logging format across all application components
- Enables structured log processing for monitoring and analysis systems
- Provides performance insights for optimization and debugging
- Configures appropriate logging levels for different environments
- Sets up log formatting suitable for production log aggregation

**Configuration Process**:
1. **Log Level Setup**: Configures root logger with specified verbosity level
2. **Formatter Selection**: Applies appropriate log formatting based on environment
3. **Handler Configuration**: Sets up console and optional file handlers
4. **Performance Monitoring**: Configures timing and resource usage logging
5. **Library Integration**: Configures third-party library logging behavior

**Usage Examples**:

#### Development Setup
```python
# Development environment with debug logging
setup_logging(
    level="DEBUG",
    format_type="console",
    enable_performance_logging=True
)
```

#### Production Setup  
```python
# Production environment with JSON logging
setup_logging(
    level="INFO", 
    format_type="json",
    enable_performance_logging=True
)
```

#### Minimal Logging Setup
```python
# Minimal logging for testing
setup_logging(
    level="ERROR",
    format_type="text", 
    enable_performance_logging=False
)
```

## Logging Format Types

### JSON Format (`format_type="json"`)

**Purpose**: Provides structured JSON logging for production environments and log aggregation systems.

**What it achieves**:
- Enables automated log parsing and analysis
- Supports structured queries in log management systems
- Provides consistent field naming across all log entries
- Facilitates integration with monitoring and alerting systems

**JSON Log Structure**:
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "modern_rag.services.embedding",
  "message": "Generated embeddings for 50 chunks",
  "module": "embedding_service",
  "function": "generate_embeddings",
  "line": 125,
  "process_id": 12345,
  "thread_name": "MainThread",
  "execution_time": 2.45,
  "memory_usage": 156.7,
  "metadata": {
    "batch_size": 32,
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_count": 50
  }
}
```

**Field Descriptions**:
- **timestamp**: ISO 8601 formatted timestamp with milliseconds
- **level**: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **logger**: Hierarchical logger name indicating source component
- **message**: Human-readable log message
- **module**: Python module name where log was generated
- **function**: Function name where log was generated
- **line**: Line number in source code
- **process_id**: Operating system process identifier
- **thread_name**: Thread name for concurrent processing tracking
- **execution_time**: Function or operation execution time (when available)
- **memory_usage**: Memory usage in MB (when performance logging enabled)
- **metadata**: Context-specific additional information

### Console Format (`format_type="console"`)

**Purpose**: Provides human-readable, color-coded logging for development environments.

**What it achieves**:
- Enhances developer productivity with readable log output
- Uses color coding to highlight different log levels
- Provides compact but informative log format
- Optimized for terminal and IDE console display

**Console Log Format**:
```
2024-01-15 10:30:45.123 | INFO     | embedding_service:125 | Generated embeddings for 50 chunks [2.45s]
2024-01-15 10:30:45.456 | WARNING  | pdf_processor:89      | Low OCR confidence: 45% for page 3
2024-01-15 10:30:45.789 | ERROR    | vector_store:234      | Failed to connect to ChromaDB: Connection refused
```

**Color Coding**:
- **DEBUG**: Gray text for debugging information
- **INFO**: White/default text for normal information
- **WARNING**: Yellow text for warning conditions
- **ERROR**: Red text for error conditions
- **CRITICAL**: Bold red text for critical errors

### Text Format (`format_type="text"`)

**Purpose**: Provides plain text logging without color codes, suitable for file output and simple processing.

**What it achieves**:
- Creates parseable plain text logs without ANSI color codes
- Suitable for file logging and simple text processing
- Maintains readability without terminal color support
- Provides consistent format across different output destinations

## Performance Logging

### Performance Monitoring Features (`enable_performance_logging=True`)

**Purpose**: Tracks system performance metrics and operation timing for optimization and monitoring.

**What it achieves**:
- Monitors function execution times for performance optimization
- Tracks memory usage patterns to identify memory leaks or excessive usage
- Provides metrics for capacity planning and resource allocation
- Enables performance regression detection across deployments

**Performance Metrics Tracked**:

#### Execution Time Monitoring
```python
# Automatic timing for key operations
logger.info(
    "Embedding generation completed",
    extra={
        "execution_time": 2.45,
        "operation": "generate_embeddings",
        "batch_size": 32
    }
)
```

#### Memory Usage Tracking
```python
# Memory usage monitoring
logger.info(
    "Document processing completed", 
    extra={
        "memory_usage": 156.7,  # MB
        "memory_delta": 23.4,   # MB increase
        "operation": "process_pdf"
    }
)
```

#### Throughput Metrics
```python
# Processing throughput tracking
logger.info(
    "Batch processing metrics",
    extra={
        "documents_processed": 100,
        "processing_time": 45.2,
        "throughput": 2.21,  # documents/second
        "average_doc_size": 12.5  # KB
    }
)
```

### Performance Log Analysis

**Key Performance Indicators (KPIs)**:
- **Processing Speed**: Documents or chunks processed per second
- **Memory Efficiency**: Memory usage per document or operation
- **Response Times**: API endpoint and function execution times
- **Resource Utilization**: CPU and memory usage patterns
- **Error Rates**: Processing failure rates and error patterns

## Integration with Application Components

### Service Integration

#### Embedding Service Logging
```python
import logging
from config.logging_config import setup_logging

# Initialize logging
setup_logging(level="INFO", format_type="json")
logger = logging.getLogger("modern_rag.services.embedding")

class EmbeddingService:
    def generate_embeddings(self, texts):
        start_time = time.time()
        logger.info("Starting embedding generation", extra={
            "text_count": len(texts),
            "model_name": self.model_name
        })
        
        try:
            embeddings = self.model.encode(texts)
            execution_time = time.time() - start_time
            
            logger.info("Embedding generation completed", extra={
                "execution_time": execution_time,
                "embeddings_shape": embeddings.shape,
                "throughput": len(texts) / execution_time
            })
            
            return embeddings
        except Exception as e:
            logger.error("Embedding generation failed", extra={
                "error": str(e),
                "text_count": len(texts)
            })
            raise
```

#### PDF Processing Logging
```python
logger = logging.getLogger("modern_rag.processing.pdf")

def extract_text_from_pdf(self, pdf_path):
    logger.info("Starting PDF processing", extra={
        "file_path": str(pdf_path),
        "file_size": os.path.getsize(pdf_path)
    })
    
    try:
        # Processing logic
        result = self._process_pdf(pdf_path)
        
        logger.info("PDF processing completed", extra={
            "pages_processed": result['metadata']['pages'],
            "text_length": len(result['text']),
            "has_scanned_pages": result['has_scanned_pages']
        })
        
        return result
    except Exception as e:
        logger.error("PDF processing failed", extra={
            "file_path": str(pdf_path),
            "error": str(e),
            "error_type": type(e).__name__
        })
        raise
```

### API Integration

#### FastAPI Request Logging
```python
import logging
from fastapi import Request
import time

logger = logging.getLogger("modern_rag.api")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    logger.info("Request started", extra={
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host
    })
    
    response = await call_next(request)
    execution_time = time.time() - start_time
    
    logger.info("Request completed", extra={
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "execution_time": execution_time
    })
    
    return response
```

### Error Handling Integration

#### Exception Logging
```python
from core.exceptions import EmbeddingError, ProcessingError

logger = logging.getLogger("modern_rag.error_handler")

def handle_embedding_error(error: EmbeddingError):
    logger.error("Embedding service error", extra={
        "error_type": type(error).__name__,
        "error_code": error.error_code,
        "error_message": str(error),
        "context": error.context,
        "traceback": traceback.format_exc()
    })
```

## Log Management and Monitoring

### Log Aggregation

**ELK Stack Integration**:
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "service": "modern-rag",
  "environment": "production",
  "version": "1.0.0",
  "logger": "modern_rag.services.embedding",
  "message": "Generated embeddings for batch",
  "metadata": {
    "batch_size": 32,
    "execution_time": 2.45
  }
}
```

**Prometheus Metrics Integration**:
```python
# Custom metrics from log data
processing_duration = Histogram('document_processing_duration_seconds')
embedding_batch_size = Histogram('embedding_batch_size')
memory_usage = Gauge('memory_usage_mb')
```

### Alerting and Monitoring

#### Error Rate Monitoring
- **High Error Rate**: Alert when error rate exceeds 5% over 5 minutes
- **Processing Failures**: Alert on repeated processing failures
- **Performance Degradation**: Alert when response times exceed thresholds

#### Performance Monitoring  
- **Memory Usage**: Alert when memory usage exceeds 80% of available
- **Processing Speed**: Alert when throughput drops below expected levels
- **Queue Depth**: Monitor processing queue lengths for bottlenecks

### Log Rotation and Retention

#### File-Based Logging (Optional)
```python
import logging.handlers

# Configure rotating file handler
file_handler = logging.handlers.RotatingFileHandler(
    'logs/modern_rag.log',
    maxBytes=100 * 1024 * 1024,  # 100MB
    backupCount=10
)
```

#### Docker Logging
```yaml
# docker-compose.yml logging configuration
services:
  modern-rag:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
```

## Configuration Examples

### Development Environment
```python
# Development setup
setup_logging(
    level="DEBUG",
    format_type="console", 
    enable_performance_logging=True
)

# Results in colored console output with detailed debugging
```

### Production Environment
```python
# Production setup
setup_logging(
    level="INFO",
    format_type="json",
    enable_performance_logging=True
)

# Results in structured JSON logs suitable for log aggregation
```

### Testing Environment
```python
# Testing setup
setup_logging(
    level="WARNING",
    format_type="text",
    enable_performance_logging=False
)

# Results in minimal logging to avoid test output pollution
```

### High-Performance Production
```python
# High-performance production setup
setup_logging(
    level="ERROR", 
    format_type="json",
    enable_performance_logging=False
)

# Results in minimal logging overhead with only error reporting
```

## Best Practices

### Logging Strategy
- **Structured Data**: Always use structured logging with context
- **Performance Impact**: Monitor logging overhead in production
- **Security**: Avoid logging sensitive data (passwords, tokens, PII)
- **Consistency**: Use consistent field names and formats across components

### Error Logging
- **Context Preservation**: Include relevant context with error logs
- **Stack Traces**: Include full stack traces for debugging
- **Error Classification**: Use appropriate log levels for different error types
- **Recovery Information**: Log recovery actions and outcomes

### Performance Logging
- **Selective Monitoring**: Monitor critical paths and bottlenecks
- **Metric Standardization**: Use consistent metrics across components
- **Baseline Establishment**: Establish performance baselines for comparison
- **Trend Analysis**: Enable trend analysis through consistent metric collection

This logging configuration provides a comprehensive, production-ready logging solution that supports both development productivity and production monitoring requirements.
