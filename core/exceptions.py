"""
Custom exception classes for the Modern RAG application.
Provides structured error handling with detailed context and user-friendly messages.
"""

from typing import Any, Dict, Optional


class ModernRAGException(Exception):
    """Base exception class for Modern RAG application."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "MODERN_RAG_ERROR",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize base exception.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for logging and debugging
            details: Additional error context and metadata
            cause: Original exception that caused this error
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# ===========================================
# Document Processing Exceptions
# ===========================================

class DocumentProcessingError(ModernRAGException):
    """Base class for document processing errors."""
    
    def __init__(self, message: str, document_path: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if document_path:
            details["document_path"] = document_path
        # Remove conflicting kwargs to avoid duplicate arguments
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_code', 'details']}
        super().__init__(message, error_code="DOCUMENT_PROCESSING_ERROR", details=details, **clean_kwargs)


class PDFExtractionError(DocumentProcessingError):
    """Error during PDF text extraction."""
    
    def __init__(self, message: str, pdf_path: str, page_number: Optional[int] = None, **kwargs):
        details = kwargs.get("details", {})
        details.update({
            "pdf_path": pdf_path,
            "page_number": page_number,
        })
        # Remove conflicting kwargs to avoid duplicate arguments
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_code', 'details']}
        super().__init__(
            message, 
            document_path=pdf_path,
            **clean_kwargs
        )


class OCRProcessingError(DocumentProcessingError):
    """Error during OCR processing of scanned documents."""
    
    def __init__(self, message: str, confidence_score: Optional[float] = None, **kwargs):
        details = kwargs.get("details", {})
        if confidence_score is not None:
            details["ocr_confidence"] = confidence_score
        # Remove conflicting kwargs to avoid duplicate arguments
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_code', 'details']}
        super().__init__(
            message,
            **clean_kwargs
        )


class ChunkingError(DocumentProcessingError):
    """Error during text chunking process."""
    
    def __init__(self, message: str, chunk_strategy: str, text_length: Optional[int] = None, **kwargs):
        details = kwargs.get("details", {})
        details.update({
            "chunk_strategy": chunk_strategy,
            "text_length": text_length,
        })
        super().__init__(
            message,
            error_code="CHUNKING_ERROR",
            details=details,
            **kwargs
        )


# ===========================================
# Embedding and Vector Store Exceptions
# ===========================================

class EmbeddingError(ModernRAGException):
    """Base class for embedding-related errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if model_name:
            details["embedding_model"] = model_name
        # Remove details from kwargs to avoid duplicate parameter
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "details"}
        super().__init__(message, error_code="EMBEDDING_ERROR", details=details, **kwargs_clean)


class ModelLoadingError(EmbeddingError):
    """Error loading embedding or language models."""
    
    def __init__(self, message: str, model_name: str, model_path: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        details.update({
            "model_name": model_name,
            "model_path": model_path,
        })
        # Remove error_code from kwargs to avoid duplicate parameter
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "error_code"}
        super().__init__(
            message,
            model_name=model_name,
            details=details,
            **kwargs_clean
        )


class VectorStoreError(ModernRAGException):
    """Base class for vector database errors."""
    
    def __init__(self, message: str, store_type: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if store_type:
            details["vector_store_type"] = store_type
        super().__init__(message, error_code="VECTOR_STORE_ERROR", details=details, **kwargs)


class VectorSearchError(VectorStoreError):
    """Error during vector similarity search."""
    
    def __init__(
        self, 
        message: str, 
        query: Optional[str] = None, 
        collection_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        details.update({
            "query": query,
            "collection_name": collection_name,
        })
        super().__init__(
            message,
            error_code="VECTOR_SEARCH_ERROR",
            details=details,
            **kwargs
        )


class VectorIndexError(VectorStoreError):
    """Error during vector indexing operations."""
    
    def __init__(
        self, 
        message: str, 
        operation: str,
        vector_count: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        details.update({
            "operation": operation,
            "vector_count": vector_count,
        })
        super().__init__(
            message,
            error_code="VECTOR_INDEX_ERROR",
            details=details,
            **kwargs
        )


# ===========================================
# Search and Retrieval Exceptions
# ===========================================

class SearchError(ModernRAGException):
    """Base class for search-related errors."""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if query:
            details["search_query"] = query
        super().__init__(message, error_code="SEARCH_ERROR", details=details, **kwargs)


class QueryProcessingError(SearchError):
    """Error during query preprocessing and enhancement."""
    
    def __init__(self, message: str, query: str, processing_step: str, **kwargs):
        details = kwargs.get("details", {})
        details.update({
            "query": query,
            "processing_step": processing_step,
        })
        super().__init__(
            message,
            error_code="QUERY_PROCESSING_ERROR",
            details=details,
            **kwargs
        )


class RetrievalError(SearchError):
    """Error during document retrieval process."""
    
    def __init__(
        self, 
        message: str, 
        retrieval_method: str,
        result_count: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        details.update({
            "retrieval_method": retrieval_method,
            "result_count": result_count,
        })
        super().__init__(
            message,
            error_code="RETRIEVAL_ERROR",
            details=details,
            **kwargs
        )


# ===========================================
# LLM and Generation Exceptions
# ===========================================

class LLMError(ModernRAGException):
    """Base class for LLM-related errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if model_name:
            details["llm_model"] = model_name
        super().__init__(message, error_code="LLM_ERROR", details=details, **kwargs)


class GenerationError(LLMError):
    """Error during text generation process."""
    
    def __init__(
        self, 
        message: str, 
        prompt_length: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        details.update({
            "prompt_length": prompt_length,
            "max_tokens": max_tokens,
        })
        super().__init__(
            message,
            error_code="GENERATION_ERROR",
            details=details,
            **kwargs
        )


class LLMConnectionError(LLMError):
    """Error connecting to LLM service (e.g., Ollama)."""
    
    def __init__(self, message: str, service_url: str, **kwargs):
        details = kwargs.get("details", {})
        details["service_url"] = service_url
        super().__init__(
            message,
            error_code="LLM_CONNECTION_ERROR",
            details=details,
            **kwargs
        )


# ===========================================
# Configuration and Validation Exceptions
# ===========================================

class ConfigurationError(ModernRAGException):
    """Configuration or settings validation error."""
    
    def __init__(self, message: str, config_field: Optional[str] = None, **kwargs):
        details = kwargs.get("details", {})
        if config_field:
            details["config_field"] = config_field
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            **kwargs
        )


class ValidationError(ModernRAGException):
    """Data validation error."""
    
    def __init__(self, message: str, field_name: str, field_value: Any = None, **kwargs):
        details = kwargs.get("details", {})
        details.update({
            "field_name": field_name,
            "field_value": str(field_value) if field_value is not None else None,
        })
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            details=details,
            **kwargs
        )


# ===========================================
# Resource and Performance Exceptions
# ===========================================

class ResourceError(ModernRAGException):
    """Resource-related errors (memory, disk, etc.)."""
    
    def __init__(self, message: str, resource_type: str, **kwargs):
        details = kwargs.get("details", {})
        details["resource_type"] = resource_type
        # Remove conflicting kwargs to avoid duplicate arguments
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_code', 'details']}
        super().__init__(
            message,
            error_code="RESOURCE_ERROR",
            details=details,
            **clean_kwargs
        )


class RAGMemoryError(ResourceError):
    """Memory-related errors."""
    
    def __init__(self, message: str, memory_usage_mb: Optional[float] = None, **kwargs):
        details = kwargs.get("details", {})
        if memory_usage_mb:
            details["memory_usage_mb"] = memory_usage_mb
        # Remove conflicting kwargs to avoid duplicate arguments
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['error_code', 'details', 'resource_type']}
        super().__init__(
            message,
            resource_type="memory",
            **clean_kwargs
        )


class TimeoutError(ModernRAGException):
    """Operation timeout error."""
    
    def __init__(self, message: str, operation: str, timeout_seconds: float, **kwargs):
        details = kwargs.get("details", {})
        details.update({
            "operation": operation,
            "timeout_seconds": timeout_seconds,
        })
        super().__init__(
            message,
            error_code="TIMEOUT_ERROR",
            details=details,
            **kwargs
        )
