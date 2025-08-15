"""
Unit tests for EmbeddingService.
Tests embedding generation, batch processing, and memory management.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time

from services.embedding_service import EmbeddingService, get_embedding_service
from core.exceptions import EmbeddingError, ModelLoadingError, RAGMemoryError


class TestEmbeddingService:
    """Test suite for EmbeddingService."""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer for testing."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.max_seq_length = 512
        mock_model.device = "cpu"
        mock_model.eval.return_value = None
        
        # Mock encode method to return appropriate embeddings
        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            # Return random embeddings of correct dimension
            return np.random.rand(len(texts), 384).astype(np.float32)
        
        mock_model.encode = mock_encode
        return mock_model
    
    @pytest.fixture
    def embedding_service(self, mock_sentence_transformer):
        """EmbeddingService instance with mocked model."""
        with patch('services.embedding_service.SentenceTransformer', return_value=mock_sentence_transformer):
            service = EmbeddingService(model_name="test-model")
            return service
    
    def test_initialization(self):
        """Test service initialization."""
        service = EmbeddingService(model_name="test-model")
        
        assert service.model_name == "test-model"
        assert not service.is_loaded
        assert service._model is None
        assert isinstance(service._embedding_stats, dict)
    
    def test_model_info_unloaded(self):
        """Test model info when model is not loaded."""
        service = EmbeddingService(model_name="test-model")
        info = service.model_info
        
        assert info['model_name'] == "test-model"
        assert info['is_loaded'] is False
        assert 'stats' in info
    
    def test_model_info_loaded(self, embedding_service):
        """Test model info when model is loaded."""
        embedding_service.load_model()
        info = embedding_service.model_info
        
        assert info['model_name'] == "test-model"
        assert info['is_loaded'] is True
        assert info['embedding_dimension'] == 384
        assert 'device' in info
    
    @patch('psutil.Process')
    def test_memory_check_normal(self, mock_process, embedding_service):
        """Test memory check under normal conditions."""
        # Mock memory usage below limit
        mock_process.return_value.memory_info.return_value.rss = 20 * (1024**3)  # 20GB
        
        memory_gb = embedding_service._check_memory_usage()
        assert memory_gb == 20.0
    
    @patch('psutil.Process')
    def test_memory_check_exceeds_limit(self, mock_process, embedding_service):
        """Test memory check when limit is exceeded."""
        # Mock memory usage above limit (default is 24GB)
        mock_process.return_value.memory_info.return_value.rss = 30 * (1024**3)  # 30GB
        
        with pytest.raises(RAGMemoryError) as exc_info:
            embedding_service._check_memory_usage()
        
        assert "Memory usage" in str(exc_info.value)
        assert exc_info.value.details['memory_usage_mb'] == 30 * 1024
    
    def test_load_model_success(self, embedding_service):
        """Test successful model loading."""
        assert not embedding_service.is_loaded
        
        embedding_service.load_model()
        
        assert embedding_service.is_loaded
        assert embedding_service._model is not None
    
    def test_load_model_already_loaded(self, embedding_service):
        """Test loading model when already loaded."""
        embedding_service.load_model()
        assert embedding_service.is_loaded
        
        # Load again - should not raise error
        embedding_service.load_model()
        assert embedding_service.is_loaded
    
    @patch('services.embedding_service.SentenceTransformer')
    def test_load_model_failure(self, mock_transformer):
        """Test model loading failure."""
        mock_transformer.side_effect = Exception("Model not found")
        
        service = EmbeddingService(model_name="invalid-model")
        
        with pytest.raises(ModelLoadingError) as exc_info:
            service.load_model()
        
        assert "Failed to load model" in str(exc_info.value)
        assert exc_info.value.details['model_name'] == "invalid-model"
    
    @pytest.mark.asyncio
    async def test_ensure_model_loaded(self, embedding_service):
        """Test async model loading."""
        assert not embedding_service.is_loaded
        
        await embedding_service.ensure_model_loaded()
        
        assert embedding_service.is_loaded
    
    def test_encode_batch_success(self, embedding_service):
        """Test successful batch encoding."""
        embedding_service.load_model()
        
        texts = ["Hello world", "Test sentence"]
        embeddings = embedding_service._encode_batch(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 384)
        assert embedding_service._embedding_stats['total_embeddings'] == 2
    
    def test_encode_batch_model_not_loaded(self, embedding_service):
        """Test batch encoding when model not loaded."""
        texts = ["Hello world"]
        
        with pytest.raises(EmbeddingError) as exc_info:
            embedding_service._encode_batch(texts)
        
        assert "Model not loaded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_embed_texts_single_string(self, embedding_service):
        """Test embedding single text string."""
        text = "Hello world"
        
        embedding = await embedding_service.embed_texts(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
    
    @pytest.mark.asyncio
    async def test_embed_texts_list(self, embedding_service):
        """Test embedding list of texts."""
        texts = ["Hello world", "Test sentence", "Another text"]
        
        embeddings = await embedding_service.embed_texts(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape == (384,) for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_embed_texts_empty_list(self, embedding_service):
        """Test embedding empty list."""
        embeddings = await embedding_service.embed_texts([])
        assert embeddings == []
    
    @pytest.mark.asyncio
    async def test_embed_texts_with_progress_callback(self, embedding_service):
        """Test embedding with progress tracking."""
        texts = ["Text 1", "Text 2", "Text 3"]
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        await embedding_service.embed_texts(texts, progress_callback=progress_callback)
        
        # Should have at least one progress call
        assert len(progress_calls) > 0
        # Final call should be (total, total)
        assert progress_calls[-1] == (3, 3)
    
    @pytest.mark.asyncio
    async def test_embed_texts_batch_processing(self, embedding_service):
        """Test batch processing with small batch size."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        
        # Use small batch size to test batching
        embeddings = await embedding_service.embed_texts(texts, batch_size=2)
        
        assert len(embeddings) == 5
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_embed_documents_success(self, embedding_service):
        """Test embedding documents with metadata."""
        documents = [
            {"content": "Document 1", "title": "Title 1", "id": "doc1"},
            {"content": "Document 2", "title": "Title 2", "id": "doc2"},
            {"content": "", "title": "Empty doc", "id": "doc3"},  # Empty content
            {"title": "No content", "id": "doc4"},  # No content field
        ]
        
        result_docs = await embedding_service.embed_documents(documents)
        
        # Should process 2 valid documents (with non-empty content)
        valid_docs = [doc for doc in result_docs if 'embedding' in doc]
        assert len(valid_docs) == 2
        
        for doc in valid_docs:
            assert 'embedding' in doc
            assert isinstance(doc['embedding'], list)
            assert len(doc['embedding']) == 384
    
    @pytest.mark.asyncio
    async def test_embed_documents_empty_list(self, embedding_service):
        """Test embedding empty document list."""
        result = await embedding_service.embed_documents([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_embed_documents_custom_text_field(self, embedding_service):
        """Test embedding documents with custom text field."""
        documents = [
            {"description": "Document 1", "id": "doc1"},
            {"description": "Document 2", "id": "doc2"},
        ]
        
        result_docs = await embedding_service.embed_documents(
            documents, text_field="description"
        )
        
        assert len(result_docs) == 2
        assert all('embedding' in doc for doc in result_docs)
    
    def test_get_embedding_dimension_loaded(self, embedding_service):
        """Test getting embedding dimension when model is loaded."""
        embedding_service.load_model()
        
        dimension = embedding_service.get_embedding_dimension()
        assert dimension == 384
    
    def test_get_embedding_dimension_not_loaded(self, embedding_service):
        """Test getting embedding dimension when model is not loaded."""
        # Should load model automatically
        dimension = embedding_service.get_embedding_dimension()
        assert dimension == 384
        assert embedding_service.is_loaded
    
    @patch('services.embedding_service.SentenceTransformer')
    def test_get_embedding_dimension_load_failure(self, mock_transformer):
        """Test getting dimension when model loading fails."""
        mock_transformer.side_effect = Exception("Load failed")
        
        service = EmbeddingService(model_name="invalid-model")
        
        with pytest.raises(ModelLoadingError):
            service.get_embedding_dimension()
    
    def test_cleanup(self, embedding_service):
        """Test service cleanup."""
        embedding_service.load_model()
        assert embedding_service.is_loaded
        
        embedding_service.cleanup()
        
        assert not embedding_service.is_loaded
        assert embedding_service._model is None
    
    def test_performance_stats_tracking(self, embedding_service):
        """Test performance statistics tracking."""
        embedding_service.load_model()
        
        # Initial stats
        stats = embedding_service._embedding_stats
        assert stats['total_embeddings'] == 0
        assert stats['total_time'] == 0.0
        
        # Process some embeddings
        texts = ["Text 1", "Text 2"]
        embedding_service._encode_batch(texts)
        
        # Check updated stats
        updated_stats = embedding_service._embedding_stats
        assert updated_stats['total_embeddings'] == 2
        assert updated_stats['total_time'] > 0
        assert updated_stats['last_batch_size'] == 2
        assert updated_stats['average_time_per_embedding'] > 0


class TestGlobalEmbeddingService:
    """Test global embedding service functions."""
    
    def test_get_embedding_service_singleton(self):
        """Test that get_embedding_service returns singleton."""
        service1 = get_embedding_service()
        service2 = get_embedding_service()
        
        assert service1 is service2
    
    @patch('services.embedding_service._embedding_service', None)
    def test_get_embedding_service_creates_new(self):
        """Test that get_embedding_service creates new instance if needed."""
        # Reset global instance
        import services.embedding_service
        services.embedding_service._embedding_service = None
        
        service = get_embedding_service()
        assert service is not None
        assert isinstance(service, EmbeddingService)


# Integration-style tests (would require actual model in real testing)
class TestEmbeddingServiceIntegration:
    """Integration tests for EmbeddingService (require model downloads)."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_real_embedding_generation(self):
        """Test with real sentence-transformers model (slow test)."""
        # This test would download and use a real model
        # Skip in regular test runs with pytest -m "not slow"
        service = EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        text = "This is a test sentence for embedding generation."
        embedding = await service.embed_texts(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # MiniLM-L6-v2 has 384 dimensions
        assert not np.all(embedding == 0)  # Should not be zero vector
        
        service.cleanup()
    
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_semantic_similarity(self):
        """Test that semantically similar texts have similar embeddings."""
        service = EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Similar sentences
        text1 = "The cat is sleeping"
        text2 = "A cat is taking a nap"
        
        # Different sentence
        text3 = "Programming languages are useful tools"
        
        embeddings = await service.embed_texts([text1, text2, text3])
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_12 = cosine_similarity(embeddings[0], embeddings[1])  # Similar sentences
        sim_13 = cosine_similarity(embeddings[0], embeddings[2])  # Different sentences
        
        # Similar sentences should have higher similarity
        assert sim_12 > sim_13
        assert sim_12 > 0.5  # Should be reasonably similar
        
        service.cleanup()
