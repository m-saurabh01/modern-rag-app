"""
Configuration management using Pydantic Settings.
Supports environment variables and .env files for all application settings.
"""

import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseSettings, Field, validator


class AppSettings(BaseSettings):
    """Main application configuration."""
    
    # Application Info
    app_name: str = Field(default="Modern RAG Application", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    app_description: str = Field(
        default="Industry-standard RAG with modern embeddings", 
        env="APP_DESCRIPTION"
    )
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=False, env="RELOAD")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ModelSettings(BaseSettings):
    """Model and embedding configuration."""
    
    # Embedding Model
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")
    
    # LLM Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3", env="OLLAMA_MODEL")
    llm_max_tokens: int = Field(default=2048, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    
    @validator("embedding_model")
    def validate_embedding_model(cls, v):
        """Validate embedding model name."""
        allowed_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        ]
        if v not in allowed_models:
            raise ValueError(f"Embedding model must be one of: {allowed_models}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ProcessingSettings(BaseSettings):
    """Document processing configuration."""
    
    # Memory Management
    max_memory_gb: int = Field(default=24, env="MAX_MEMORY_GB")
    processing_batch_size: int = Field(default=50, env="PROCESSING_BATCH_SIZE")
    max_concurrent_documents: int = Field(default=10, env="MAX_CONCURRENT_DOCUMENTS")
    
    # Text Processing
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    min_chunk_size: int = Field(default=100, env="MIN_CHUNK_SIZE")
    max_chunks_per_document: int = Field(default=500, env="MAX_CHUNKS_PER_DOCUMENT")
    
    # PDF Processing
    enable_ocr: bool = Field(default=True, env="ENABLE_OCR")
    ocr_language: str = Field(default="eng", env="OCR_LANGUAGE")
    pdf_dpi: int = Field(default=300, env="PDF_DPI")
    max_pdf_size_mb: int = Field(default=100, env="MAX_PDF_SIZE_MB")
    
    @validator("chunk_overlap")
    def validate_overlap(cls, v, values):
        """Ensure overlap is less than chunk size."""
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class VectorDBSettings(BaseSettings):
    """Vector database configuration."""
    
    # Database Type
    vector_db_type: Literal["chromadb", "qdrant"] = Field(
        default="chromadb", env="VECTOR_DB_TYPE"
    )
    
    # ChromaDB Configuration
    chromadb_host: str = Field(default="localhost", env="CHROMADB_HOST")
    chromadb_port: int = Field(default=8000, env="CHROMADB_PORT")
    chromadb_persist_directory: str = Field(
        default="./storage/chromadb", env="CHROMADB_PERSIST_DIRECTORY"
    )
    chromadb_collection_name: str = Field(
        default="document_chunks", env="CHROMADB_COLLECTION_NAME"
    )
    
    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection_name: str = Field(
        default="document_chunks", env="QDRANT_COLLECTION_NAME"
    )
    qdrant_vector_size: int = Field(default=384, env="QDRANT_VECTOR_SIZE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class StorageSettings(BaseSettings):
    """Storage and file handling configuration."""
    
    # Storage Paths
    document_storage_path: Path = Field(
        default=Path("./storage/documents"), env="DOCUMENT_STORAGE_PATH"
    )
    processed_storage_path: Path = Field(
        default=Path("./storage/processed"), env="PROCESSED_STORAGE_PATH"
    )
    temp_storage_path: Path = Field(
        default=Path("./storage/temp"), env="TEMP_STORAGE_PATH"
    )
    backup_storage_path: Path = Field(
        default=Path("./storage/backups"), env="BACKUP_STORAGE_PATH"
    )
    
    # File Limits
    max_upload_size_mb: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")
    allowed_file_types: List[str] = Field(
        default=["pdf", "txt", "md", "docx"], env="ALLOWED_FILE_TYPES"
    )
    
    @validator("*", pre=True)
    def parse_paths(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str) and any(keyword in cls.__fields__ for keyword in ["path"]):
            return Path(v)
        return v
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        for path_field in ["document_storage_path", "processed_storage_path", 
                          "temp_storage_path", "backup_storage_path"]:
            path = getattr(self, path_field)
            path.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class SearchSettings(BaseSettings):
    """Search and retrieval configuration."""
    
    # Search Limits
    default_search_limit: int = Field(default=10, env="DEFAULT_SEARCH_LIMIT")
    max_search_limit: int = Field(default=100, env="MAX_SEARCH_LIMIT")
    search_score_threshold: float = Field(default=0.5, env="SEARCH_SCORE_THRESHOLD")
    
    # Hybrid Search
    enable_hybrid_search: bool = Field(default=True, env="ENABLE_HYBRID_SEARCH")
    keyword_search_weight: float = Field(default=0.3, env="KEYWORD_SEARCH_WEIGHT")
    semantic_search_weight: float = Field(default=0.7, env="SEMANTIC_SEARCH_WEIGHT")
    
    @validator("keyword_search_weight", "semantic_search_weight")
    def validate_weights(cls, v, values):
        """Ensure search weights sum to 1.0."""
        if "keyword_search_weight" in values:
            total = v + values.get("semantic_search_weight", 0.7)
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError("Search weights must sum to 1.0")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class LoggingSettings(BaseSettings):
    """Logging and monitoring configuration."""
    
    # Log Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", env="LOG_LEVEL"
    )
    log_format: Literal["json", "text"] = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default="./logs/app.log", env="LOG_FILE")
    log_max_size: str = Field(default="10MB", env="LOG_MAX_SIZE")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Monitoring
    enable_metrics: bool = Field(default=False, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class Settings:
    """Aggregated application settings."""
    
    def __init__(self):
        self.app = AppSettings()
        self.model = ModelSettings()
        self.processing = ProcessingSettings()
        self.vector_db = VectorDBSettings()
        self.storage = StorageSettings()
        self.search = SearchSettings()
        self.logging = LoggingSettings()


# Global settings instance
settings = Settings()
