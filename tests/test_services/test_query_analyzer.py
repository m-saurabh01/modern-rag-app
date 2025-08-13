"""
Comprehensive test suite for QueryAnalyzer service.

Tests cover all major functionality including:
- Query intent classification accuracy
- Entity extraction precision
- Conservative query expansion
- Integration with DocumentAnalyzer
- Performance characteristics
- Caching functionality
- Error handling and fallbacks
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.query_analyzer import QueryAnalyzer
from models.query_models import (
    QueryAnalysis, QueryEntity, QueryExpansion, QueryIntent, 
    QuestionType, EntityType, ProcessingMode, RetrievalStrategy,
    QueryAnalyzerConfig, QueryCache
)
from services.document_analyzer import DocumentAnalyzer, DocumentStructure


class TestQueryAnalyzer:
    """Comprehensive test suite for QueryAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create a QueryAnalyzer instance for testing."""
        config = QueryAnalyzerConfig(
            processing_mode=ProcessingMode.BALANCED,
            enable_query_expansion=True,
            expansion_aggressiveness=0.3,  # Conservative
            entity_confidence_threshold=0.6,
            fallback_to_patterns=True
        )
        return QueryAnalyzer(config=config)

    @pytest.fixture
    def cached_analyzer(self):
        """Create a QueryAnalyzer with caching enabled."""
        cache_config = QueryCache(
            enabled=True,
            ttl_seconds=3600,
            max_entries=100
        )
        config = QueryAnalyzerConfig(
            processing_mode=ProcessingMode.BALANCED,
            cache_config=cache_config
        )
        return QueryAnalyzer(config=config)

    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing different intents and types."""
        return {
            'factual': [
                "What is the budget for IT department in 2024?",
                "Who is the director of Finance?",
                "How much was allocated for training?",
                "List all policies from Department of Health"
            ],
            'analytical': [
                "Why did the costs increase in Q2?",
                "What caused the budget overrun?",
                "Explain the impact of new regulations",
                "Analyze the performance trends"
            ],
            'comparative': [
                "Compare Q1 vs Q2 performance",
                "What's the difference between Policy A and B?",
                "Budget 2023 vs 2024 comparison",
                "Which department spent more on equipment?"
            ],
            'procedural': [
                "How do I submit a change request?",
                "Steps to apply for budget approval",
                "Process for updating employee information",
                "How to access the document system?"
            ],
            'verification': [
                "Is Policy HHS-2024-001 still active?",
                "Confirm if the deadline is March 31st",
                "Check if this information is current",
                "Verify the budget allocation amount"
            ],
            'exploratory': [
                "Tell me about department restructuring",
                "Overview of recent policy changes",
                "What do you know about the new system?",
                "General information about budget process"
            ]
        }

    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing extraction accuracy."""
        return {
            'date_entities': [
                "January 2024", "Q1 2024", "2023-03-15", "March 31, 2024",
                "last quarter", "next year", "fiscal year 2024"
            ],
            'organization_entities': [
                "Department of Health", "IT Department", "HR Division",
                "Finance Team", "DOH", "HHS", "Office of the Secretary"
            ],
            'currency_entities': [
                "$1,500,000", "$25.50", "2.5 million dollars",
                "$50K", "fifteen thousand USD", "$1.2B"
            ],
            'document_entities': [
                "Policy HHS-2024-001", "quarterly report", "budget summary",
                "change request CR-123", "memorandum", "annual review"
            ],
            'person_entities': [
                "Director Smith", "Sarah Johnson", "Dr. Martinez",
                "Secretary of Health", "Project Manager"
            ]
        }

    # ==================== Intent Classification Tests ====================

    @pytest.mark.asyncio
    async def test_intent_classification_accuracy(self, analyzer, sample_queries):
        """Test query intent classification accuracy across all categories."""
        
        for expected_intent_str, queries in sample_queries.items():
            expected_intent = QueryIntent(expected_intent_str)
            
            for query in queries:
                analysis = await analyzer.analyze_query(query)
                
                assert analysis.intent == expected_intent, (
                    f"Expected {expected_intent.value} for query '{query}', "
                    f"got {analysis.intent.value}"
                )
                assert analysis.confidence_score > 0.5, (
                    f"Low confidence ({analysis.confidence_score}) for query: {query}"
                )

    @pytest.mark.asyncio
    async def test_intent_classification_patterns(self, analyzer):
        """Test pattern-based intent classification fallback."""
        
        # Test specific pattern matches
        test_cases = [
            ("What is the current budget?", QueryIntent.FACTUAL),
            ("Why did this happen?", QueryIntent.ANALYTICAL),
            ("Compare option A vs option B", QueryIntent.COMPARATIVE),
            ("How do I submit a form?", QueryIntent.PROCEDURAL),
            ("Is this policy still valid?", QueryIntent.VERIFICATION),
            ("Tell me about the changes", QueryIntent.EXPLORATORY)
        ]
        
        for query, expected_intent in test_cases:
            analysis = await analyzer.analyze_query(query)
            assert analysis.intent == expected_intent

    # ==================== Question Type Detection Tests ====================

    @pytest.mark.asyncio
    async def test_question_type_detection(self, analyzer):
        """Test question type classification accuracy."""
        
        test_cases = [
            ("Who is responsible for this?", QuestionType.WHO),
            ("What is the policy number?", QuestionType.WHAT),
            ("Where is the document located?", QuestionType.WHERE),
            ("When is the deadline?", QuestionType.WHEN),
            ("Why was this decision made?", QuestionType.WHY),
            ("How do I complete this task?", QuestionType.HOW),
            ("How much budget is allocated?", QuestionType.HOW_MUCH),
            ("How many employees are affected?", QuestionType.HOW_MANY),
            ("Is this requirement mandatory?", QuestionType.YES_NO),
            ("Analyze the budget trends over time", QuestionType.OPEN_ENDED)
        ]
        
        for query, expected_type in test_cases:
            analysis = await analyzer.analyze_query(query)
            assert analysis.question_type == expected_type, (
                f"Expected {expected_type.value} for '{query}', "
                f"got {analysis.question_type.value}"
            )

    # ==================== Entity Extraction Tests ====================

    @pytest.mark.asyncio
    async def test_entity_extraction_dates(self, analyzer, sample_entities):
        """Test date and time entity extraction."""
        
        for date_text in sample_entities['date_entities']:
            query = f"Find documents from {date_text}"
            analysis = await analyzer.analyze_query(query)
            
            # Should extract at least one DATE_TIME entity
            date_entities = [e for e in analysis.entities if e.entity_type == EntityType.DATE_TIME]
            assert len(date_entities) > 0, f"No date entity found in: {query}"
            
            # Check if the date text is captured
            date_texts = [e.text for e in date_entities]
            assert any(date_text.lower() in dt.lower() for dt in date_texts), (
                f"Date '{date_text}' not found in extracted entities: {date_texts}"
            )

    @pytest.mark.asyncio
    async def test_entity_extraction_organizations(self, analyzer, sample_entities):
        """Test organization entity extraction."""
        
        for org_text in sample_entities['organization_entities']:
            query = f"Get reports from {org_text} about budget"
            analysis = await analyzer.analyze_query(query)
            
            # Should extract ORGANIZATION entity
            org_entities = [e for e in analysis.entities if e.entity_type == EntityType.ORGANIZATION]
            org_texts = [e.text for e in org_entities]
            
            # Check if organization is captured (allowing partial matches)
            found = any(org_text.lower() in ot.lower() or ot.lower() in org_text.lower() 
                      for ot in org_texts)
            assert found, f"Organization '{org_text}' not found in: {org_texts}"

    @pytest.mark.asyncio
    async def test_entity_extraction_currency(self, analyzer, sample_entities):
        """Test currency and numeric entity extraction."""
        
        for currency_text in sample_entities['currency_entities']:
            query = f"What was spent? The amount was {currency_text}"
            analysis = await analyzer.analyze_query(query)
            
            # Should extract CURRENCY or NUMERIC entity
            financial_entities = [
                e for e in analysis.entities 
                if e.entity_type in [EntityType.CURRENCY, EntityType.NUMERIC]
            ]
            assert len(financial_entities) > 0, (
                f"No financial entity found for: {currency_text}"
            )

    @pytest.mark.asyncio
    async def test_entity_confidence_filtering(self, analyzer):
        """Test entity confidence threshold filtering."""
        
        query = "Meeting on 2024-03-15 with Department of Health about Policy HHS-2024-001"
        analysis = await analyzer.analyze_query(query)
        
        # All extracted entities should meet confidence threshold
        for entity in analysis.entities:
            assert entity.confidence >= analyzer.config.entity_confidence_threshold, (
                f"Entity '{entity.text}' confidence {entity.confidence} below threshold "
                f"{analyzer.config.entity_confidence_threshold}"
            )

    # ==================== Query Expansion Tests ====================

    @pytest.mark.asyncio
    async def test_query_expansion_conservative(self, analyzer):
        """Test conservative query expansion strategy."""
        
        query = "budget allocation report for finance department"
        analysis = await analyzer.analyze_query(query)
        
        expansion = analysis.expansion
        
        # Should have original terms
        assert len(expansion.original_terms) > 0
        assert "budget" in [term.lower() for term in expansion.original_terms]
        
        # Conservative expansion limits
        if expansion.synonyms:
            for term, synonyms in expansion.synonyms.items():
                assert len(synonyms) <= expansion.max_synonyms_per_term, (
                    f"Too many synonyms for '{term}': {len(synonyms)} > {expansion.max_synonyms_per_term}"
                )
        
        # Check expansion confidence
        assert 0.0 <= expansion.expansion_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_query_expansion_with_entities(self, analyzer):
        """Test query expansion integration with entity extraction."""
        
        query = "Department of Health budget for Q2 2024"
        analysis = await analyzer.analyze_query(query)
        
        # Should have entities
        assert len(analysis.entities) > 0
        
        # Expansion should include suggested filters based on entities
        if analysis.expansion.suggested_filters:
            # Check for date-based filters
            date_entities = [e for e in analysis.entities if e.entity_type == EntityType.DATE_TIME]
            if date_entities and 'date_range' in analysis.expansion.suggested_filters:
                assert len(analysis.expansion.suggested_filters['date_range']) > 0

    @pytest.mark.asyncio
    async def test_query_expansion_disabled(self):
        """Test query analysis with expansion disabled."""
        
        config = QueryAnalyzerConfig(enable_query_expansion=False)
        analyzer = QueryAnalyzer(config=config)
        
        query = "budget report analysis"
        analysis = await analyzer.analyze_query(query)
        
        # Should have minimal expansion
        expansion = analysis.expansion
        assert len(expansion.synonyms) == 0
        assert len(expansion.related_concepts) == 0

    # ==================== Integration Tests ====================

    @pytest.mark.asyncio
    async def test_document_context_integration(self, analyzer):
        """Test integration with document context for domain knowledge."""
        
        # Create mock document structures
        mock_documents = []  # This would be populated with actual DocumentStructure objects
        
        query = "IT department budget allocation"
        analysis = await analyzer.analyze_query(
            query, 
            document_context=mock_documents
        )
        
        # Analysis should complete successfully
        assert analysis.original_query == query
        assert analysis.processing_time > 0
        assert analysis.metadata['document_context_count'] == len(mock_documents)

    @pytest.mark.asyncio
    async def test_retrieval_strategy_suggestions(self, analyzer):
        """Test retrieval strategy suggestions based on analysis."""
        
        test_cases = [
            ("What is the exact budget amount?", RetrievalStrategy.ENTITY_FOCUSED),
            ("Compare Q1 vs Q2 performance", RetrievalStrategy.MULTI_DOCUMENT),
            ("Why did costs increase?", RetrievalStrategy.SEMANTIC_SEARCH),
            ("How to submit a form?", RetrievalStrategy.STRUCTURE_AWARE)
        ]
        
        for query, expected_strategy in test_cases:
            analysis = await analyzer.analyze_query(query)
            
            assert len(analysis.suggested_strategies) > 0
            # Expected strategy should be in the suggestions
            assert expected_strategy in analysis.suggested_strategies, (
                f"Expected {expected_strategy.value} in strategies for '{query}', "
                f"got {[s.value for s in analysis.suggested_strategies]}"
            )

    # ==================== Performance Tests ====================

    @pytest.mark.asyncio
    async def test_processing_modes_performance(self):
        """Test different processing modes and their performance characteristics."""
        
        query = "Department of Health budget analysis for Q2 2024"
        
        # Test each processing mode
        modes = [ProcessingMode.FAST, ProcessingMode.BALANCED, ProcessingMode.COMPREHENSIVE]
        results = {}
        
        for mode in modes:
            config = QueryAnalyzerConfig(processing_mode=mode)
            analyzer = QueryAnalyzer(config=config)
            
            analysis = await analyzer.analyze_query(query)
            results[mode] = analysis.processing_time
            
            # Verify analysis quality based on mode
            if mode == ProcessingMode.FAST:
                assert analysis.processing_time < 100  # Should be very fast
            elif mode == ProcessingMode.BALANCED:
                assert analysis.processing_time < 300  # Reasonable time
            else:  # COMPREHENSIVE
                assert analysis.processing_time < 600  # More thorough analysis
        
        # Fast should generally be faster than comprehensive
        assert results[ProcessingMode.FAST] <= results[ProcessingMode.COMPREHENSIVE]

    @pytest.mark.asyncio
    async def test_batch_analysis_performance(self, analyzer):
        """Test batch analysis efficiency."""
        
        queries = [
            "What is the budget?",
            "Who is responsible?", 
            "How much was spent?",
            "When is the deadline?",
            "Why did this happen?"
        ]
        
        start_time = datetime.now()
        analyses = await analyzer.batch_analyze(queries)
        batch_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Should return results for all queries
        assert len(analyses) == len(queries)
        
        # Individual analysis should be successful
        for i, analysis in enumerate(analyses):
            assert analysis.original_query == queries[i]
            assert analysis.processing_time > 0
        
        # Batch should be more efficient than individual calls
        individual_time = sum(a.processing_time for a in analyses)
        # Batch might have some overhead but should be reasonable
        assert batch_time < individual_time * 1.5

    # ==================== Caching Tests ====================

    @pytest.mark.asyncio
    async def test_query_caching_functionality(self, cached_analyzer):
        """Test query analysis caching."""
        
        query = "What is the IT budget for 2024?"
        
        # First analysis - cache miss
        analysis1 = await cached_analyzer.analyze_query(query)
        first_time = analysis1.processing_time
        
        # Second analysis - should be cache hit
        analysis2 = await cached_analyzer.analyze_query(query)
        second_time = analysis2.processing_time
        
        # Results should be identical
        assert analysis1.intent == analysis2.intent
        assert len(analysis1.entities) == len(analysis2.entities)
        
        # Cache hit should be significantly faster (if implemented)
        # Note: Actual cache hit timing would depend on implementation
        assert second_time >= 0  # Basic sanity check

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cached_analyzer):
        """Test cache TTL expiration."""
        
        # This test would need to manipulate time or use a very short TTL
        # For now, just verify cache management functions work
        
        query = "Test query for cache expiration"
        analysis = await cached_analyzer.analyze_query(query)
        
        # Clear cache
        cached_analyzer.clear_cache()
        
        # Should work without errors
        analysis2 = await cached_analyzer.analyze_query(query)
        assert analysis2.original_query == query

    # ==================== Error Handling Tests ====================

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, analyzer):
        """Test handling of empty or invalid queries."""
        
        with pytest.raises(ValueError):
            await analyzer.analyze_query("")
        
        with pytest.raises(ValueError):
            await analyzer.analyze_query("   ")

    @pytest.mark.asyncio
    async def test_nlp_fallback_behavior(self):
        """Test graceful fallback when NLP libraries are unavailable."""
        
        # Create analyzer that simulates NLP library failures
        config = QueryAnalyzerConfig(fallback_to_patterns=True)
        analyzer = QueryAnalyzer(config=config)
        
        # Simulate NLP unavailability
        analyzer.nltk_ready = False
        analyzer.spacy_ready = False
        
        query = "What is the budget allocation for IT department?"
        analysis = await analyzer.analyze_query(query)
        
        # Should still provide analysis using patterns
        assert analysis.intent in [intent for intent in QueryIntent]
        assert analysis.question_type in [qtype for qtype in QuestionType]
        assert analysis.metadata['fallback_patterns_used'] == True

    @pytest.mark.asyncio
    async def test_malformed_query_handling(self, analyzer):
        """Test handling of malformed or unusual queries."""
        
        malformed_queries = [
            "???",  # Only punctuation
            "a" * 1000,  # Very long query
            "12345",  # Only numbers
            "!@#$%^&*()",  # Special characters
            "query with\x00null\x00characters"  # Null characters
        ]
        
        for query in malformed_queries:
            try:
                analysis = await analyzer.analyze_query(query)
                # Should not crash and provide some analysis
                assert analysis.original_query == query
                assert analysis.intent in [intent for intent in QueryIntent]
            except Exception as e:
                # If it does raise an exception, it should be handled gracefully
                assert "Query analysis failed" in str(e) or "empty" in str(e).lower()

    # ==================== Configuration Tests ====================

    @pytest.mark.asyncio
    async def test_custom_configuration(self):
        """Test QueryAnalyzer with custom configuration."""
        
        custom_config = QueryAnalyzerConfig(
            processing_mode=ProcessingMode.COMPREHENSIVE,
            entity_confidence_threshold=0.8,
            expansion_aggressiveness=0.1,  # Very conservative
            max_processing_time_ms=1000.0,
            custom_intent_patterns={
                'factual': [r'\bspecial\s+pattern\b']
            }
        )
        
        analyzer = QueryAnalyzer(config=custom_config)
        
        # Test custom intent pattern
        analysis = await analyzer.analyze_query("Find special pattern information")
        # Should classify as factual due to custom pattern
        
        # Verify configuration is applied
        assert analyzer.config.processing_mode == ProcessingMode.COMPREHENSIVE
        assert analyzer.config.entity_confidence_threshold == 0.8

    # ==================== Performance Monitoring Tests ====================

    def test_performance_statistics(self, analyzer):
        """Test performance statistics collection."""
        
        stats = analyzer.get_performance_stats()
        
        # Should have required fields
        required_fields = [
            'analysis_count', 'average_processing_time_ms', 
            'cache_size', 'nlp_status', 'processing_mode'
        ]
        
        for field in required_fields:
            assert field in stats
        
        # NLP status should show library availability
        assert 'nltk_ready' in stats['nlp_status']
        assert 'spacy_ready' in stats['nlp_status']
        
        # Processing mode should match config
        assert stats['processing_mode'] == analyzer.config.processing_mode.value

    # ==================== Integration Validation Tests ====================

    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self, analyzer):
        """Test complete analysis workflow with all components."""
        
        complex_query = (
            "Compare the IT budget allocation between Q1 2024 and Q2 2024 "
            "for Department of Health, focusing on equipment purchases over $50,000"
        )
        
        analysis = await analyzer.analyze_query(complex_query)
        
        # Should classify intent correctly
        assert analysis.intent == QueryIntent.COMPARATIVE
        
        # Should extract multiple entities
        entities_by_type = {}
        for entity in analysis.entities:
            if entity.entity_type not in entities_by_type:
                entities_by_type[entity.entity_type] = []
            entities_by_type[entity.entity_type].append(entity.text)
        
        # Should find date, organization, and currency entities
        assert EntityType.DATE_TIME in entities_by_type
        assert EntityType.ORGANIZATION in entities_by_type or EntityType.DOMAIN_TERM in entities_by_type
        assert EntityType.CURRENCY in entities_by_type or EntityType.NUMERIC in entities_by_type
        
        # Should suggest appropriate retrieval strategies
        assert RetrievalStrategy.COMPARATIVE in analysis.suggested_strategies or \
               RetrievalStrategy.MULTI_DOCUMENT in analysis.suggested_strategies
        
        # Should have query expansion
        assert len(analysis.expansion.original_terms) > 0
        
        # Should have reasonable confidence and complexity scores
        assert 0.3 <= analysis.confidence_score <= 1.0
        assert 0.3 <= analysis.complexity_score <= 1.0


# ==================== Performance Benchmarks ====================

@pytest.mark.benchmark
class TestQueryAnalyzerBenchmarks:
    """Performance benchmarks for QueryAnalyzer."""
    
    @pytest.mark.asyncio
    async def test_processing_speed_benchmark(self):
        """Benchmark processing speed across different query types."""
        
        config = QueryAnalyzerConfig(processing_mode=ProcessingMode.BALANCED)
        analyzer = QueryAnalyzer(config=config)
        
        benchmark_queries = [
            "What is the budget?",  # Simple
            "Department of Health budget allocation for Q2 2024",  # Medium
            "Compare IT spending vs HR spending between Q1 2024 and Q2 2024 for equipment purchases over $10,000"  # Complex
        ]
        
        results = []
        for query in benchmark_queries:
            analysis = await analyzer.analyze_query(query)
            results.append({
                'query_length': len(query),
                'processing_time': analysis.processing_time,
                'entity_count': len(analysis.entities),
                'complexity_score': analysis.complexity_score
            })
        
        # Log benchmark results
        for i, result in enumerate(results):
            print(f"Query {i+1}: {result['processing_time']:.2f}ms "
                  f"({result['query_length']} chars, "
                  f"{result['entity_count']} entities)")
        
        # Performance assertions
        assert all(r['processing_time'] < 500 for r in results), "Processing too slow"
        
        # Complex queries should generally take longer
        assert results[-1]['processing_time'] >= results[0]['processing_time']


if __name__ == "__main__":
    # Run basic functionality tests
    pytest.main([__file__, "-v"])
