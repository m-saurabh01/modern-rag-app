"""
Structured logging configuration using structlog.
Provides JSON-formatted logs with correlation IDs and performance metrics.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .settings import settings


def setup_logging() -> None:
    """
    Configure structured logging for the application.
    
    Sets up:
    - JSON formatting for production logs
    - Rich formatting for development console output
    - File rotation with size limits
    - Correlation ID tracking
    - Performance timing
    """
    
    # Create logs directory if it doesn't exist
    if settings.logging.log_file:
        log_path = Path(settings.logging.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            _add_correlation_id,
            _add_performance_info,
            structlog.processors.JSONRenderer() if settings.logging.log_format == "json" 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.logging.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.logging.log_level.upper()),
        format="%(message)s",
        handlers=_get_handlers(),
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def _get_handlers() -> list:
    """Get configured log handlers based on settings."""
    handlers = []
    
    # Console handler with rich formatting for development
    if settings.app.debug:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_level=True,
            show_path=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(logging.DEBUG)
        handlers.append(console_handler)
    else:
        # Simple stream handler for production
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(getattr(logging, settings.logging.log_level.upper()))
        handlers.append(stream_handler)
    
    # File handler with rotation
    if settings.logging.log_file:
        max_bytes = _parse_size(settings.logging.log_max_size)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=settings.logging.log_file,
            maxBytes=max_bytes,
            backupCount=settings.logging.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        handlers.append(file_handler)
    
    return handlers


def _parse_size(size_str: str) -> int:
    """Parse size string like '10MB' to bytes."""
    size_str = size_str.upper().strip()
    if size_str.endswith("KB"):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith("MB"):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith("GB"):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)  # Assume bytes


def _add_correlation_id(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add correlation ID to log events."""
    import contextvars
    
    # Try to get correlation ID from context
    correlation_id = contextvars.copy_context().get("correlation_id", None)
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    
    return event_dict


def _add_performance_info(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add performance metrics to log events."""
    import time
    import psutil
    import os
    
    # Add timestamp and process info
    event_dict["timestamp"] = time.time()
    event_dict["process_id"] = os.getpid()
    
    # Add memory usage (in MB)
    try:
        process = psutil.Process()
        event_dict["memory_mb"] = round(process.memory_info().rss / 1024 / 1024, 2)
        event_dict["cpu_percent"] = round(process.cpu_percent(), 2)
    except Exception:
        # Don't fail logging if psutil fails
        pass
    
    return event_dict


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name or "modern_rag_app")


# Performance timing decorator
def log_performance(operation_name: str):
    """
    Decorator to log performance metrics for functions.
    
    Args:
        operation_name: Name of the operation for logging
        
    Example:
        @log_performance("document_processing")
        async def process_document(doc):
            # ... processing logic
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"{operation_name} completed successfully",
                    operation=operation_name,
                    duration_seconds=round(duration, 3),
                    function=func.__name__,
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{operation_name} failed",
                    operation=operation_name,
                    duration_seconds=round(duration, 3),
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"{operation_name} completed successfully",
                    operation=operation_name,
                    duration_seconds=round(duration, 3),
                    function=func.__name__,
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{operation_name} failed",
                    operation=operation_name,
                    duration_seconds=round(duration, 3),
                    function=func.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
