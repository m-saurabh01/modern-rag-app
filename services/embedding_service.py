"""
Modern embedding service using sentence-transformers.
Replaces TF-IDF with dense semantic embeddings for superior search quality.
"""

import asyncio
from typing import List, Optional, Dict, Any, Union
import numpy as np
from pathlib import Path
import time
import psutil
import threading
from sentence_transformers import SentenceTransformer
import torch

from config import settings, get_logger, log_performance
from core.exceptions import EmbeddingError, ModelLoadingError, MemoryError


class EmbeddingService:
    """
    Production-ready embedding service with batch processing and memory optimization.
    
    Features:
    - Sentence-transformer models for semantic embeddings
    - Batch processing for efficiency
    - Memory monitoring and limits
    - Model caching and warm-up
    - Async-friendly design with thread pool
    - Progress tracking for large batches
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding service with specified model.
        
        Args:
            model_name: Sentence transformer model name. 
                       Defaults to settings.model.embedding_model
        
        Raises:
            ModelLoadingError: If model fails to load
            MemoryError: If insufficient memory for model
        """
        self.model_name = model_name or settings.model.embedding_model
        self.batch_size = settings.model.embedding_batch_size
        self.logger = get_logger(__name__)
        
        # Model state
        self._model: Optional[SentenceTransformer] = None
        self._model_lock = threading.Lock()
        self._is_loading = False
        self._load_start_time: Optional[float] = None
        
        # Performance tracking
        self._embedding_stats = {
            'total_embeddings': 0,
            'total_time': 0.0,
            'average_time_per_embedding': 0.0,
            'last_batch_size': 0,
            'last_batch_time': 0.0
        }
        
        self.logger.info(
            "EmbeddingService initialized",
            model_name=self.model_name,
            batch_size=self.batch_size
        )
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model is not None and not self._is_loading
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        info = {
            'model_name': self.model_name,
            'is_loaded': self.is_loaded,
            'batch_size': self.batch_size,
            'stats': self._embedding_stats.copy()
        }
        
        if self._model is not None:
            info.update({
                'embedding_dimension': self._model.get_sentence_embedding_dimension(),
                'max_sequence_length': getattr(self._model, 'max_seq_length', None),
                'device': str(self._model.device) if hasattr(self._model, 'device') else 'cpu'
            })
        
        return info
    
    def _check_memory_usage(self) -> float:
        """
        Check current memory usage in GB.
        
        Returns:
            Current memory usage in GB
            
        Raises:
            MemoryError: If memory usage exceeds configured limit
        """
        memory_gb = psutil.Process().memory_info().rss / (1024**3)
        
        if memory_gb > settings.processing.max_memory_gb:
            raise MemoryError(
                f"Memory usage {memory_gb:.2f}GB exceeds limit {settings.processing.max_memory_gb}GB",
                memory_usage_mb=memory_gb * 1024
            )
        
        return memory_gb
    
    @log_performance("model_loading")
    def load_model(self) -> None:
        """
        Load the sentence transformer model with memory monitoring.
        
        Raises:
            ModelLoadingError: If model loading fails
            MemoryError: If insufficient memory
        """
        if self.is_loaded:
            self.logger.debug("Model already loaded", model_name=self.model_name)
            return
        
        with self._model_lock:
            if self._model is not None:  # Double-check after acquiring lock
                return
            
            self._is_loading = True
            self._load_start_time = time.time()
            
            try:
                self.logger.info("Loading embedding model", model_name=self.model_name)
                
                # Check memory before loading
                initial_memory = self._check_memory_usage()
                
                # Load model with optimizations
                self._model = SentenceTransformer(
                    self.model_name,
                    device='cpu'  # CPU-only for our constraint
                )
                
                # Optimize for inference
                self._model.eval()
                
                # Check memory after loading
                final_memory = self._check_memory_usage()
                memory_used = final_memory - initial_memory
                
                # Warm up the model with a test embedding
                test_text = "This is a test sentence for model warm-up."
                _ = self._model.encode([test_text], batch_size=1, show_progress_bar=False)
                
                load_time = time.time() - self._load_start_time
                
                self.logger.info(
                    "Model loaded successfully",
                    model_name=self.model_name,
                    load_time_seconds=round(load_time, 2),
                    memory_used_mb=round(memory_used * 1024, 2),
                    embedding_dimension=self._model.get_sentence_embedding_dimension()
                )
                
            except Exception as e:
                self.logger.error(
                    "Failed to load embedding model",
                    model_name=self.model_name,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise ModelLoadingError(
                    f"Failed to load model '{self.model_name}': {str(e)}",
                    model_name=self.model_name,
                    cause=e
                )
            
            finally:
                self._is_loading = False
                self._load_start_time = None
    
    async def ensure_model_loaded(self) -> None:
        """
        Ensure model is loaded asynchronously.
        
        Uses thread pool to avoid blocking the event loop during model loading.
        """
        if self.is_loaded:
            return
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.load_model)
    
    def _encode_batch(
        self, 
        texts: List[str], 
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode a batch of texts to embeddings.
        
        Args:
            texts: List of texts to encode
            normalize_embeddings: Whether to normalize embeddings to unit length
            
        Returns:
            Numpy array of embeddings with shape (len(texts), embedding_dim)
            
        Raises:
            EmbeddingError: If encoding fails
        """
        if not self.is_loaded:
            raise EmbeddingError("Model not loaded", model_name=self.model_name)
        
        try:
            start_time = time.time()
            
            # Generate embeddings
            embeddings = self._model.encode(
                texts,
                batch_size=min(len(texts), self.batch_size),
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Update statistics
            batch_time = time.time() - start_time
            self._embedding_stats['total_embeddings'] += len(texts)
            self._embedding_stats['total_time'] += batch_time
            self._embedding_stats['last_batch_size'] = len(texts)
            self._embedding_stats['last_batch_time'] = batch_time
            self._embedding_stats['average_time_per_embedding'] = (
                self._embedding_stats['total_time'] / self._embedding_stats['total_embeddings']
            )
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(
                f"Failed to encode batch of {len(texts)} texts: {str(e)}",
                model_name=self.model_name,
                details={'batch_size': len(texts)},
                cause=e
            )
    
    @log_performance("text_embedding")
    async def embed_texts(
        self, 
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        normalize_embeddings: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s) with batch processing and progress tracking.
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Override default batch size
            normalize_embeddings: Whether to normalize embeddings to unit length
            progress_callback: Optional callback for progress updates (current, total)
            
        Returns:
            Single embedding array for string input, list of arrays for list input
            
        Raises:
            EmbeddingError: If embedding generation fails
            MemoryError: If memory limit exceeded
        """
        # Ensure model is loaded
        await self.ensure_model_loaded()
        
        # Handle single text input
        if isinstance(texts, str):
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self._encode_batch, [texts], normalize_embeddings
            )
            return embeddings[0]
        
        # Handle batch processing
        if not texts:
            return []
        
        effective_batch_size = batch_size or self.batch_size
        embeddings = []
        total_texts = len(texts)
        
        self.logger.info(
            "Starting batch embedding",
            total_texts=total_texts,
            batch_size=effective_batch_size
        )
        
        for i in range(0, total_texts, effective_batch_size):
            batch_texts = texts[i:i + effective_batch_size]
            batch_num = i // effective_batch_size + 1
            total_batches = (total_texts + effective_batch_size - 1) // effective_batch_size
            
            # Progress callback
            if progress_callback:
                progress_callback(i, total_texts)
            
            # Check memory before processing batch
            self._check_memory_usage()
            
            # Process batch in thread pool
            batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self._encode_batch, batch_texts, normalize_embeddings
            )
            
            embeddings.extend(batch_embeddings)
            
            self.logger.debug(
                "Processed embedding batch",
                batch_num=batch_num,
                total_batches=total_batches,
                batch_size=len(batch_texts)
            )
        
        # Final progress callback
        if progress_callback:
            progress_callback(total_texts, total_texts)
        
        self.logger.info(
            "Batch embedding completed",
            total_texts=total_texts,
            total_embeddings=len(embeddings)
        )
        
        return embeddings
    
    async def embed_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = 'content',
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Embed documents with metadata preservation.
        
        Args:
            documents: List of document dictionaries
            text_field: Field name containing text to embed
            progress_callback: Optional progress callback
            
        Returns:
            Documents with added 'embedding' field
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not documents:
            return []
        
        # Extract texts
        texts = []
        valid_documents = []
        
        for doc in documents:
            if text_field in doc and doc[text_field]:
                texts.append(str(doc[text_field]))
                valid_documents.append(doc)
        
        if not texts:
            self.logger.warning("No valid texts found in documents", text_field=text_field)
            return documents
        
        self.logger.info(
            "Embedding documents",
            total_documents=len(valid_documents),
            text_field=text_field
        )
        
        # Generate embeddings
        embeddings = await self.embed_texts(
            texts, 
            progress_callback=progress_callback
        )
        
        # Add embeddings to documents
        for doc, embedding in zip(valid_documents, embeddings):
            doc['embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        return valid_documents
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            Embedding dimension
            
        Raises:
            ModelLoadingError: If model is not loaded
        """
        if not self.is_loaded:
            # Try to load model if not loaded
            try:
                self.load_model()
            except Exception as e:
                raise ModelLoadingError(
                    "Cannot get embedding dimension: model not loaded",
                    model_name=self.model_name,
                    cause=e
                )
        
        return self._model.get_sentence_embedding_dimension()
    
    def cleanup(self) -> None:
        """Clean up model and free memory."""
        with self._model_lock:
            if self._model is not None:
                # Clear model from memory
                del self._model
                self._model = None
                
                # Force garbage collection
                import gc
                gc.collect()
                
                self.logger.info(
                    "EmbeddingService cleaned up",
                    model_name=self.model_name
                )


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get the global embedding service instance.
    
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    
    return _embedding_service
