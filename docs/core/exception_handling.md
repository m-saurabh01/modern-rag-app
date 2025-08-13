# Exception Handling System Documentation

## Overview

The Modern RAG application implements a comprehensive exception handling system that provides structured error management, detailed context, and user-friendly error messages. The system is built around a hierarchical exception class structure that enables precise error categorization and effective debugging.

## Architecture

### Base Exception Class

#### ModernRAGException
The foundation of the exception hierarchy, providing common functionality for all application errors.

**Key Features:**
- Structured error codes for categorization
- Detailed error context and metadata
- Cause tracking for error chaining
- API-friendly error serialization

```python
class ModernRAGException(Exception):
    def __init__(
        self,
        message: str,
        error_code: str = "MODERN_RAG_ERROR",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
```

## Exception Categories

### 1. Document Processing Exceptions

#### DocumentProcessingError
Base class for document-related processing errors.

**Specialized Exceptions:**
- `PDFExtractionError`: PDF text extraction failures
- `OCRProcessingError`: OCR processing issues with confidence tracking
- `ChunkingError`: Text chunking failures with strategy context

**Usage Example:**
```python
try:
    text = await pdf_processor.extract_text(pdf_path)
except PDFExtractionError as e:
    logger.error(f"PDF extraction failed: {e.message}")
    logger.error(f"PDF: {e.details['pdf_path']}")
    logger.error(f"Page: {e.details.get('page_number', 'unknown')}")
```

### 2. Embedding and Vector Store Exceptions

#### EmbeddingError
Base class for embedding-related operations.

**Specialized Exceptions:**
- `ModelLoadingError`: Embedding/LLM model loading failures
- `VectorStoreError`: Vector database operation errors
- `VectorSearchError`: Search operation failures
- `VectorIndexError`: Indexing operation errors

**Usage Example:**
```python
try:
    embeddings = await embedding_service.embed_texts(texts)
except ModelLoadingError as e:
    logger.error(f"Model loading failed: {e.message}")
    logger.error(f"Model: {e.details['model_name']}")
    logger.error(f"Path: {e.details.get('model_path', 'default')}")
```

### 3. Search and Retrieval Exceptions

#### SearchError
Base class for search-related operations.

**Specialized Exceptions:**
- `QueryProcessingError`: Query preprocessing failures
- `RetrievalError`: Document retrieval errors

**Usage Example:**
```python
try:
    results = await search_service.search(query)
except QueryProcessingError as e:
    logger.error(f"Query processing failed: {e.message}")
    logger.error(f"Query: {e.details['query']}")
    logger.error(f"Step: {e.details['processing_step']}")
```

### 4. LLM and Generation Exceptions

#### LLMError
Base class for language model operations.

**Specialized Exceptions:**
- `GenerationError`: Text generation failures with token context
- `LLMConnectionError`: Service connection errors (e.g., Ollama)

**Usage Example:**
```python
try:
    response = await llm_service.generate(prompt)
except LLMConnectionError as e:
    logger.error(f"LLM connection failed: {e.message}")
    logger.error(f"Service URL: {e.details['service_url']}")
```

### 5. Configuration and Validation Exceptions

#### ConfigurationError
Configuration and settings validation errors.

#### ValidationError
Data validation errors with field context.

**Usage Example:**
```python
try:
    config = load_configuration(config_file)
except ConfigurationError as e:
    logger.error(f"Configuration error: {e.message}")
    logger.error(f"Field: {e.details.get('config_field', 'unknown')}")
```

### 6. Resource and Performance Exceptions

#### ResourceError
Resource-related errors (memory, disk, etc.).

**Specialized Exceptions:**
- `MemoryError`: Memory-related issues with usage tracking
- `TimeoutError`: Operation timeout errors with timing context

**Usage Example:**
```python
try:
    result = await process_large_document(document)
except MemoryError as e:
    logger.error(f"Memory error: {e.message}")
    logger.error(f"Usage: {e.details.get('memory_usage_mb')}MB")
```

## Error Context and Metadata

### Structured Error Details

Each exception includes detailed context information:

```python
{
    "error": True,
    "error_code": "PDF_EXTRACTION_ERROR",
    "message": "Failed to extract text from PDF page",
    "details": {
        "pdf_path": "/path/to/document.pdf",
        "page_number": 5,
        "file_size_mb": 15.2,
        "processing_time_ms": 1500
    }
}
```

### Error Code Categories

#### Document Processing
- `DOCUMENT_PROCESSING_ERROR`: General document processing
- `PDF_EXTRACTION_ERROR`: PDF-specific extraction issues
- `OCR_PROCESSING_ERROR`: OCR processing failures
- `CHUNKING_ERROR`: Text chunking problems

#### Embedding and Storage
- `EMBEDDING_ERROR`: General embedding issues
- `MODEL_LOADING_ERROR`: Model loading failures
- `VECTOR_STORE_ERROR`: Vector database issues
- `VECTOR_SEARCH_ERROR`: Search operation failures
- `VECTOR_INDEX_ERROR`: Indexing operation failures

#### Search and Retrieval
- `SEARCH_ERROR`: General search issues
- `QUERY_PROCESSING_ERROR`: Query preprocessing failures
- `RETRIEVAL_ERROR`: Document retrieval problems

#### LLM Operations
- `LLM_ERROR`: General language model issues
- `GENERATION_ERROR`: Text generation failures
- `LLM_CONNECTION_ERROR`: Service connection problems

#### System and Configuration
- `CONFIGURATION_ERROR`: Configuration validation
- `VALIDATION_ERROR`: Data validation issues
- `RESOURCE_ERROR`: Resource management
- `MEMORY_ERROR`: Memory-related problems
- `TIMEOUT_ERROR`: Operation timeouts

## Usage Patterns

### Exception Handling in Services

```python
async def process_document(self, document_path: str) -> ProcessedDocument:
    """Process document with comprehensive error handling."""
    try:
        # Document processing logic
        return processed_document
    
    except PDFExtractionError as e:
        # Log with context
        self.logger.error(
            f"PDF extraction failed for {document_path}",
            extra={
                "error_code": e.error_code,
                "details": e.details,
                "document_path": document_path
            }
        )
        # Re-raise or handle appropriately
        raise
    
    except Exception as e:
        # Wrap unexpected errors
        raise DocumentProcessingError(
            message=f"Unexpected error processing document: {str(e)}",
            document_path=document_path,
            cause=e
        )
```

### API Error Responses

```python
from fastapi import HTTPException

@router.post("/process-document")
async def process_document_endpoint(file_path: str):
    try:
        result = await document_processor.process(file_path)
        return {"success": True, "result": result}
    
    except ModernRAGException as e:
        # Convert to HTTP response
        raise HTTPException(
            status_code=500,
            detail=e.to_dict()
        )
    
    except Exception as e:
        # Handle unexpected errors
        error = ModernRAGException(
            message="Unexpected server error",
            error_code="INTERNAL_SERVER_ERROR",
            details={"original_error": str(e)}
        )
        raise HTTPException(status_code=500, detail=error.to_dict())
```

### Error Recovery Patterns

```python
async def robust_embedding_service(texts: List[str]) -> List[Vector]:
    """Embedding service with fallback strategies."""
    try:
        return await primary_embedding_service.embed_texts(texts)
    
    except ModelLoadingError as e:
        self.logger.warning(f"Primary model failed: {e.message}")
        try:
            return await fallback_embedding_service.embed_texts(texts)
        except EmbeddingError as fallback_error:
            # Combine error context
            raise EmbeddingError(
                message="Both primary and fallback embedding services failed",
                details={
                    "primary_error": e.to_dict(),
                    "fallback_error": fallback_error.to_dict()
                }
            )
```

## Integration with Logging

### Structured Logging

```python
import structlog

logger = structlog.get_logger(__name__)

try:
    result = await process_operation()
except ModernRAGException as e:
    logger.error(
        "Operation failed",
        error_code=e.error_code,
        error_message=e.message,
        error_details=e.details,
        operation="process_operation"
    )
```

### Error Correlation

```python
import uuid
from contextvars import ContextVar

correlation_id: ContextVar[str] = ContextVar('correlation_id')

async def process_with_correlation(data):
    correlation_id.set(str(uuid.uuid4()))
    
    try:
        return await process_data(data)
    except ModernRAGException as e:
        # Add correlation ID to error details
        e.details["correlation_id"] = correlation_id.get()
        logger.error(f"Processing failed", extra=e.to_dict())
        raise
```

## Testing Exception Handling

### Unit Tests for Exceptions

```python
import pytest
from core.exceptions import PDFExtractionError

def test_pdf_extraction_error():
    """Test PDF extraction error creation and serialization."""
    error = PDFExtractionError(
        message="Failed to extract text",
        pdf_path="/test/document.pdf",
        page_number=5
    )
    
    assert error.error_code == "PDF_EXTRACTION_ERROR"
    assert error.details["pdf_path"] == "/test/document.pdf"
    assert error.details["page_number"] == 5
    
    # Test serialization
    error_dict = error.to_dict()
    assert error_dict["error"] == True
    assert error_dict["error_code"] == "PDF_EXTRACTION_ERROR"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_service_error_handling():
    """Test service error handling and propagation."""
    service = DocumentProcessingService()
    
    with pytest.raises(PDFExtractionError) as exc_info:
        await service.process_pdf("/nonexistent/file.pdf")
    
    error = exc_info.value
    assert "nonexistent/file.pdf" in error.details["pdf_path"]
    assert error.error_code == "PDF_EXTRACTION_ERROR"
```

## Monitoring and Alerting

### Error Metrics Collection

```python
from prometheus_client import Counter, Histogram

error_counter = Counter(
    'rag_errors_total',
    'Total number of RAG application errors',
    ['error_code', 'service']
)

error_duration = Histogram(
    'rag_error_handling_duration_seconds',
    'Time spent handling errors'
)

def handle_exception(e: ModernRAGException, service_name: str):
    error_counter.labels(
        error_code=e.error_code,
        service=service_name
    ).inc()
```

### Error Alerting

```python
async def critical_error_alert(error: ModernRAGException):
    """Send alerts for critical errors."""
    critical_codes = {
        "MEMORY_ERROR",
        "LLM_CONNECTION_ERROR", 
        "VECTOR_STORE_ERROR"
    }
    
    if error.error_code in critical_codes:
        await alert_service.send_alert(
            title=f"Critical RAG Error: {error.error_code}",
            message=error.message,
            details=error.details
        )
```

## Best Practices

### Exception Design
1. **Specific Error Types**: Create specific exception classes for different error scenarios
2. **Rich Context**: Include relevant details for debugging and recovery
3. **Error Codes**: Use consistent, searchable error codes
4. **Cause Chains**: Preserve original exception information

### Error Handling
1. **Fail Fast**: Catch and handle errors as close to the source as possible
2. **Context Preservation**: Maintain error context through the call stack
3. **Graceful Degradation**: Implement fallback strategies where appropriate
4. **User-Friendly Messages**: Provide clear, actionable error messages

### Logging and Monitoring
1. **Structured Logging**: Use structured logs for better analysis
2. **Error Correlation**: Implement correlation IDs for request tracking
3. **Metrics Collection**: Track error rates and patterns
4. **Alerting**: Set up alerts for critical error conditions

### Testing
1. **Exception Coverage**: Test both success and failure paths
2. **Error Serialization**: Verify error objects serialize correctly
3. **Integration Testing**: Test error propagation through system layers
4. **Recovery Testing**: Verify error recovery and fallback mechanisms
