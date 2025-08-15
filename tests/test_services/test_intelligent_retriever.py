"""
Test suite for IntelligentRetriever service.

Comprehensive tests for Phase 3.3d implementation including all performance modes,
re-ranking complexity levels, multi-modal search, and switching capabilities.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from services.intelligent_retriever import IntelligentRetriever
from models.retrieval_models import (
    RetrievalConfig, RetrievalResult, RankedChunk, MultiModalResult,
    RetrievalMode, RerankingComplexity, ContextExpansion,
    create_speed_optimized_config, create_balanced_config, create_accuracy_optimized_config
)
from models.query_models import QueryAnalysis, QueryIntent, QueryEntity, EntityType
from services.chunking_service import TextChunk


class TestIntelligentRetriever:
    """Test suite for IntelligentRetriever service."""
    
    @pytest.fixture
    async def mock_dependencies(self):
        """Create mock dependencies for IntelligentRetriever."""
        vector_store = Mock()
        embedding_service = Mock()
        document_analyzer = Mock()
        chunking_service = Mock()
        
        # Mock async methods
        vector_store.similarity_search = AsyncMock()
        vector_store.similarity_search_with_filters = AsyncMock()
        embedding_service.generate_embedding = AsyncMock()
        
        return {
            'vector_store': vector_store,
            'embedding_service': embedding_service,
            'document_analyzer': document_analyzer,
            'chunking_service': chunking_service
        }
    
    
    @pytest.fixture
    def sample_query_analysis(self):
        """Create sample query analysis for testing."""
        return QueryAnalysis(
            original_query="What is the IT budget for 2024?",
            intent=QueryIntent.FACTUAL,
            entities=[
                QueryEntity(
                    text="IT",
                    entity_type=EntityType.ORGANIZATION,
                    start_pos=12,
                    end_pos=14,
                    confidence=0.9
                ),
                QueryEntity(
                    text="2024",
                    entity_type=EntityType.DATE_TIME,
                    start_pos=25,
                    end_pos=29,
                    confidence=0.95
                )
            ],
            confidence_score=0.85,
            processing_time_ms=150.0
        )
    
    
    @pytest.fixture
    def sample_text_chunks(self):
        """Create sample text chunks for testing."""
        return [
            TextChunk(
                content="The IT budget for fiscal year 2024 is allocated at $2.5 million.",
                metadata={
                    'document_id': 'doc1',
                    'chunk_id': 'chunk1',
                    'section_type': 'paragraph',
                    'document_type': 'budget',
                    'overlapping_entities': [{'text': 'IT', 'type': 'ORGANIZATION'}]
                },
                start_pos=0,
                end_pos=66,
                chunk_type='text'
            ),
            TextChunk(
                content="Previous year IT spending was $2.2 million, showing 13% increase.",
                metadata={
                    'document_id': 'doc1',
                    'chunk_id': 'chunk2',
                    'section_type': 'paragraph',
                    'document_type': 'budget'
                },
                start_pos=67,
                end_pos=133,
                chunk_type='text'
            )
        ]
    
    
    async def test_intelligent_retriever_initialization(self, mock_dependencies):
        """Test IntelligentRetriever initialization."""
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service'],
            document_analyzer=mock_dependencies['document_analyzer']
        )
        
        assert retriever.vector_store is not None
        assert retriever.embedding_service is not None
        assert retriever.config.retrieval_mode == RetrievalMode.BALANCED
        assert retriever.config.reranking_complexity == RerankingComplexity.ADVANCED
        assert retriever.stats.total_searches == 0
    
    
    async def test_performance_mode_switching(self, mock_dependencies):
        """Test switching between performance modes."""
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        # Test speed mode switch
        speed_config = retriever.switch_performance_mode(RetrievalMode.SPEED_OPTIMIZED)
        assert speed_config.retrieval_mode == RetrievalMode.SPEED_OPTIMIZED
        assert speed_config.target_response_time_ms == 300.0
        assert speed_config.reranking_complexity == RerankingComplexity.BASIC
        
        # Test accuracy mode switch
        accuracy_config = retriever.switch_performance_mode(RetrievalMode.ACCURACY_OPTIMIZED)
        assert accuracy_config.retrieval_mode == RetrievalMode.ACCURACY_OPTIMIZED
        assert accuracy_config.target_response_time_ms == 1000.0
        assert accuracy_config.reranking_complexity == RerankingComplexity.COMPREHENSIVE
        
        # Test balanced mode switch
        balanced_config = retriever.switch_performance_mode(RetrievalMode.BALANCED)
        assert balanced_config.retrieval_mode == RetrievalMode.BALANCED
        assert balanced_config.target_response_time_ms == 500.0
        assert balanced_config.reranking_complexity == RerankingComplexity.ADVANCED
    
    
    async def test_reranking_complexity_switching(self, mock_dependencies):
        """Test switching between re-ranking complexity levels."""
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        # Test basic complexity
        basic_config = retriever.switch_reranking_complexity(RerankingComplexity.BASIC)
        assert basic_config.reranking_complexity == RerankingComplexity.BASIC
        
        # Test comprehensive complexity
        comprehensive_config = retriever.switch_reranking_complexity(RerankingComplexity.COMPREHENSIVE)
        assert comprehensive_config.reranking_complexity == RerankingComplexity.COMPREHENSIVE
        
        # Test advanced complexity
        advanced_config = retriever.switch_reranking_complexity(RerankingComplexity.ADVANCED)
        assert advanced_config.reranking_complexity == RerankingComplexity.ADVANCED
    
    
    @patch('time.time')
    async def test_retrieve_intelligent_basic(self, mock_time, mock_dependencies, sample_query_analysis, sample_text_chunks):
        """Test basic intelligent retrieval functionality."""
        mock_time.return_value = 1000.0
        
        # Setup mocks
        mock_dependencies['vector_store'].similarity_search.return_value = [
            Mock(document=Mock(content=chunk.content, metadata=chunk.metadata), score=0.85)
            for chunk in sample_text_chunks
        ]
        mock_dependencies['embedding_service'].generate_embedding.return_value = [0.1] * 384
        
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        # Mock internal methods for testing
        retriever._convert_to_text_chunk = Mock(side_effect=lambda doc: sample_text_chunks[0])
        retriever._calculate_adaptive_weights = Mock(return_value=Mock(
            text_weight=0.5, table_weight=0.3, entity_weight=0.2, structure_weight=0.0
        ))
        
        result = await retriever.retrieve_intelligent(
            query_analysis=sample_query_analysis,
            top_k=5
        )
        
        assert isinstance(result, RetrievalResult)
        assert result.query_analysis == sample_query_analysis
        assert len(result.ranked_chunks) <= 5
        assert result.confidence_score >= 0.0
        assert result.processing_time > 0
        
        # Verify statistics updated
        assert retriever.stats.total_searches > 0
    
    
    async def test_query_adaptive_weighting(self, mock_dependencies):
        """Test query-adaptive weighting calculation."""
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        config = create_balanced_config()
        
        # Test factual query (should boost tables and entities)
        factual_analysis = Mock(
            intent=QueryIntent.FACTUAL,
            entities=[Mock(entity_type=EntityType.CURRENCY)]
        )
        
        weights = retriever._calculate_adaptive_weights(factual_analysis, config)
        
        assert weights.table_weight > 0.25  # Should be boosted from default 25%
        assert weights.entity_weight > 0.15  # Should be boosted from default 15%
        
        # Test analytical query (should boost text and structure)
        analytical_analysis = Mock(
            intent=QueryIntent.ANALYTICAL,
            entities=[]
        )
        
        weights = retriever._calculate_adaptive_weights(analytical_analysis, config)
        
        assert weights.text_weight > 0.6   # Should be boosted from default 60%
        assert weights.structure_weight > 0.0  # Should get some weight
        
        # Test comparative query (should strongly boost tables)
        comparative_analysis = Mock(
            intent=QueryIntent.COMPARATIVE,
            entities=[]
        )
        
        weights = retriever._calculate_adaptive_weights(comparative_analysis, config)
        
        assert weights.table_weight > 0.4  # Should be significantly boosted
    
    
    async def test_multi_modal_search(self, mock_dependencies, sample_query_analysis):
        """Test multi-modal search functionality."""
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        # Mock the individual search modes
        retriever._search_text_mode = AsyncMock(return_value=[Mock(spec=RankedChunk)])
        retriever._search_table_mode = AsyncMock(return_value=[Mock()])
        retriever._search_entity_mode = AsyncMock(return_value=[Mock()])
        retriever._search_structure_mode = AsyncMock(return_value=[Mock()])
        retriever._fuse_multi_modal_results = Mock(return_value=[Mock(spec=RankedChunk)])
        retriever._calculate_fusion_confidence = Mock(return_value=0.85)
        
        result = await retriever.search_multi_modal(sample_query_analysis)
        
        assert isinstance(result, MultiModalResult)
        assert result.fusion_confidence > 0.0
        assert len(result.combined_results) > 0
        
        # Verify all search modes were called
        retriever._search_text_mode.assert_called_once()
        retriever._search_table_mode.assert_called_once()
        retriever._search_entity_mode.assert_called_once()
    
    
    async def test_structure_aware_search(self, mock_dependencies):
        """Test structure-aware search functionality."""
        # Setup mock search results
        mock_search_result = Mock(
            document=Mock(
                metadata={
                    'document_type': 'budget',
                    'section_type': 'table',
                    'section_level': 1
                }
            ),
            score=0.85
        )
        
        mock_dependencies['vector_store'].similarity_search_with_filters.return_value = [mock_search_result]
        mock_dependencies['embedding_service'].generate_embedding.return_value = [0.1] * 384
        
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        # Mock internal methods
        retriever._build_structure_filters = Mock(return_value={'document_type': 'budget'})
        retriever._calculate_structure_relevance = Mock(return_value=0.8)
        retriever._convert_to_text_chunk = Mock(return_value=Mock())
        retriever._calculate_section_alignment = Mock(return_value=0.7)
        retriever._calculate_hierarchy_bonus = Mock(return_value=0.1)
        retriever._calculate_document_type_match = Mock(return_value=0.9)
        
        structure_filters = {
            'document_type': 'budget',
            'section_type': 'table'
        }
        
        results = await retriever.search_by_structure(
            query="IT budget allocation",
            structure_filters=structure_filters
        )
        
        assert len(results) > 0
        for result in results:
            assert hasattr(result, 'structure_relevance_score')
            assert result.structure_relevance_score > 0.3
    
    
    async def test_performance_characteristics(self, mock_dependencies, sample_query_analysis):
        """Test performance characteristics of different modes."""
        
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        # Mock dependencies to return quickly
        retriever._balanced_search = AsyncMock(return_value=Mock(
            combined_results=[Mock(spec=RankedChunk)],
            total_candidates=10
        ))
        retriever._dynamic_rerank = Mock(return_value=[Mock(spec=RankedChunk)])
        retriever._expand_context = AsyncMock(return_value=[Mock(spec=RankedChunk)])
        retriever._calculate_retrieval_confidence = Mock(return_value=0.85)
        retriever._estimate_precision = Mock(return_value=0.80)
        
        # Test balanced mode performance
        start_time = time.time()
        result = await retriever.retrieve_intelligent(sample_query_analysis, top_k=10)
        elapsed = (time.time() - start_time) * 1000
        
        # Should complete within reasonable time (allowing for test overhead)
        assert elapsed < 2000  # 2 second max for test environment
        assert result.processing_time > 0
        
        # Test speed mode
        speed_config = create_speed_optimized_config()
        
        start_time = time.time()
        speed_result = await retriever.retrieve_intelligent(
            sample_query_analysis, 
            config_override=speed_config
        )
        speed_elapsed = (time.time() - start_time) * 1000
        
        # Speed mode should be faster (in theory)
        assert speed_result.processing_time > 0
    
    
    async def test_re_ranking_complexity_levels(self, mock_dependencies, sample_text_chunks):
        """Test different re-ranking complexity levels."""
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        # Create mock ranked chunks
        ranked_chunks = [
            RankedChunk(
                chunk=chunk,
                base_similarity_score=0.85,
                structure_relevance_score=0.0,
                entity_alignment_score=0.0,
                intent_alignment_score=0.0,
                context_coherence_score=0.0,
                ranking_factors_used=[]
            )
            for chunk in sample_text_chunks
        ]
        
        query_analysis = Mock(
            intent=QueryIntent.FACTUAL,
            entities=[Mock(entity_type=EntityType.ORGANIZATION)]
        )
        
        # Mock internal scoring methods
        retriever._calculate_intent_alignment = Mock(return_value=0.7)
        retriever._calculate_entity_alignment_score = Mock(return_value=0.8)
        retriever._calculate_structure_relevance_score = Mock(return_value=0.6)
        retriever._calculate_context_coherence = Mock(return_value=0.5)
        retriever._calculate_authority_score = Mock(return_value=0.4)
        retriever._calculate_recency_score = Mock(return_value=0.3)
        retriever._calculate_cross_reference_score = Mock(return_value=0.2)
        retriever._calculate_temporal_relevance = Mock(return_value=0.1)
        
        # Test basic complexity
        basic_config = create_balanced_config()
        basic_config.reranking_complexity = RerankingComplexity.BASIC
        
        basic_result = retriever._dynamic_rerank(ranked_chunks.copy(), query_analysis, basic_config)
        
        assert len(basic_result) == len(ranked_chunks)
        for chunk in basic_result:
            assert chunk.final_rank_score > 0
            assert len(chunk.ranking_factors_used) >= 2  # At least similarity + intent
        
        # Test comprehensive complexity
        comprehensive_config = create_balanced_config()
        comprehensive_config.reranking_complexity = RerankingComplexity.COMPREHENSIVE
        
        comprehensive_result = retriever._dynamic_rerank(ranked_chunks.copy(), query_analysis, comprehensive_config)
        
        assert len(comprehensive_result) == len(ranked_chunks)
        for chunk in comprehensive_result:
            assert chunk.final_rank_score > 0
            assert len(chunk.ranking_factors_used) > 6  # Should have more factors
    
    
    async def test_caching_functionality(self, mock_dependencies, sample_query_analysis):
        """Test caching functionality."""
        config = create_balanced_config()
        config.enable_caching = True
        
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service'],
            config=config
        )
        
        # Mock the search process
        retriever._balanced_search = AsyncMock(return_value=Mock(
            combined_results=[Mock(spec=RankedChunk)],
            total_candidates=5
        ))
        retriever._dynamic_rerank = Mock(return_value=[Mock(spec=RankedChunk)])
        retriever._expand_context = AsyncMock(return_value=[Mock(spec=RankedChunk)])
        retriever._calculate_retrieval_confidence = Mock(return_value=0.85)
        retriever._estimate_precision = Mock(return_value=0.80)
        
        # First call should miss cache
        result1 = await retriever.retrieve_intelligent(sample_query_analysis)
        assert retriever.stats.cache_misses == 1
        assert retriever.stats.cache_hits == 0
        
        # Second identical call should hit cache
        result2 = await retriever.retrieve_intelligent(sample_query_analysis)
        assert retriever.stats.cache_hits == 1
        
        # Results should be equivalent
        assert result1.query_analysis == result2.query_analysis
    
    
    async def test_error_handling(self, mock_dependencies, sample_query_analysis):
        """Test error handling and graceful fallbacks."""
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        # Test vector store failure
        mock_dependencies['vector_store'].similarity_search.side_effect = Exception("Vector store error")
        
        with pytest.raises(Exception):  # Should raise RetrievalError
            await retriever.retrieve_intelligent(sample_query_analysis)
        
        # Verify error count increased
        assert retriever.stats.error_count > 0
        
        # Test embedding service failure
        mock_dependencies['embedding_service'].generate_embedding.side_effect = Exception("Embedding error")
        
        with pytest.raises(Exception):
            await retriever.search_by_structure(
                query="test query",
                structure_filters={'document_type': 'test'}
            )
    
    
    async def test_performance_statistics(self, mock_dependencies, sample_query_analysis):
        """Test performance statistics tracking."""
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        # Mock search process
        retriever._balanced_search = AsyncMock(return_value=Mock(
            combined_results=[Mock(spec=RankedChunk)],
            total_candidates=5
        ))
        retriever._dynamic_rerank = Mock(return_value=[Mock(spec=RankedChunk)])
        retriever._expand_context = AsyncMock(return_value=[Mock(spec=RankedChunk)])
        retriever._calculate_retrieval_confidence = Mock(return_value=0.85)
        retriever._estimate_precision = Mock(return_value=0.80)
        
        # Execute several searches
        for _ in range(3):
            await retriever.retrieve_intelligent(sample_query_analysis)
        
        # Check statistics
        stats = retriever.get_performance_stats()
        
        assert stats['total_searches'] == 3
        assert stats['average_response_time_ms'] > 0
        assert stats['current_mode'] == 'balanced'
        assert stats['current_complexity'] == 'advanced'
        assert stats['error_count'] == 0
        
        # Test statistics reset
        retriever.reset_statistics()
        
        stats_after_reset = retriever.get_performance_stats()
        assert stats_after_reset['total_searches'] == 0
        assert stats_after_reset['error_count'] == 0


class TestRetrievalConfigurations:
    """Test retrieval configuration classes and helpers."""
    
    def test_speed_optimized_config(self):
        """Test speed-optimized configuration."""
        config = create_speed_optimized_config()
        
        assert config.retrieval_mode == RetrievalMode.SPEED_OPTIMIZED
        assert config.target_response_time_ms == 300.0
        assert config.reranking_complexity == RerankingComplexity.BASIC
        assert config.structure_influence_weight == 0.15  # Lower for speed
        assert config.context_expansion == ContextExpansion.MINIMAL
        assert not config.enable_semantic_table_analysis  # Disabled for speed
    
    
    def test_balanced_config(self):
        """Test balanced configuration (default)."""
        config = create_balanced_config()
        
        assert config.retrieval_mode == RetrievalMode.BALANCED
        assert config.target_response_time_ms == 500.0
        assert config.reranking_complexity == RerankingComplexity.ADVANCED
        assert config.structure_influence_weight == 0.25  # Moderate influence
        assert config.context_expansion == ContextExpansion.DOCUMENT_CONTEXT
        assert config.enable_semantic_table_analysis
    
    
    def test_accuracy_optimized_config(self):
        """Test accuracy-optimized configuration."""
        config = create_accuracy_optimized_config()
        
        assert config.retrieval_mode == RetrievalMode.ACCURACY_OPTIMIZED
        assert config.target_response_time_ms == 1000.0
        assert config.reranking_complexity == RerankingComplexity.COMPREHENSIVE
        assert config.structure_influence_weight == 0.3  # Higher for accuracy
        assert config.context_expansion == ContextExpansion.CROSS_DOCUMENT
        assert config.enable_cross_document_context
        assert config.enable_semantic_table_analysis
        assert config.table_cell_analysis


# Benchmark tests for performance validation
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    async def test_speed_mode_performance(self, mock_dependencies, sample_query_analysis):
        """Benchmark speed mode performance."""
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        # Mock for fast completion
        retriever._speed_optimized_search = AsyncMock(return_value=Mock(
            combined_results=[Mock(spec=RankedChunk)] * 5,
            total_candidates=10
        ))
        retriever._dynamic_rerank = Mock(return_value=[Mock(spec=RankedChunk)] * 5)
        retriever._calculate_retrieval_confidence = Mock(return_value=0.85)
        retriever._estimate_precision = Mock(return_value=0.80)
        
        speed_config = create_speed_optimized_config()
        
        start_time = time.time()
        result = await retriever.retrieve_intelligent(
            sample_query_analysis,
            config_override=speed_config
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should be under target time (allowing for test overhead)
        assert result.processing_time < 300  # Internal processing time
        print(f"Speed mode completed in {elapsed_ms:.1f}ms")
    
    
    @pytest.mark.benchmark
    async def test_accuracy_mode_performance(self, mock_dependencies, sample_query_analysis):
        """Benchmark accuracy mode performance."""
        retriever = IntelligentRetriever(
            vector_store=mock_dependencies['vector_store'],
            embedding_service=mock_dependencies['embedding_service']
        )
        
        # Mock comprehensive processing
        retriever._accuracy_optimized_search = AsyncMock(return_value=Mock(
            combined_results=[Mock(spec=RankedChunk)] * 10,
            total_candidates=20
        ))
        retriever._dynamic_rerank = Mock(return_value=[Mock(spec=RankedChunk)] * 10)
        retriever._expand_context = AsyncMock(return_value=[Mock(spec=RankedChunk)] * 10)
        retriever._calculate_retrieval_confidence = Mock(return_value=0.92)
        retriever._estimate_precision = Mock(return_value=0.88)
        
        accuracy_config = create_accuracy_optimized_config()
        
        start_time = time.time()
        result = await retriever.retrieve_intelligent(
            sample_query_analysis,
            config_override=accuracy_config
        )
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should be under target time (allowing for test overhead)
        assert result.processing_time < 1000  # Internal processing time
        print(f"Accuracy mode completed in {elapsed_ms:.1f}ms")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
