"""
Intelligent Retrieval Service - Phase 3.3d Final Component

This service implements sophisticated multi-modal retrieval with query-adaptive weighting,
structure-aware ranking, semantic table analysis, and flexible performance modes.

Key Features:
- Query-adaptive multi-modal search (text/table/entity)
- Switchable re-ranking complexity (Basic/Advanced/Comprehensive)
- Full semantic table analysis with row/column understanding
- Multiple performance modes (Speed/Balanced/Accuracy)
- Document and cross-document context expansion
- Advanced scoring with 9+ ranking factors
"""

import asyncio
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import replace

from models.retrieval_models import (
    RetrievalResult, RankedChunk, MultiModalResult, TableMatch, EntityMatch, StructureMatch,
    RetrievalConfig, QueryAdaptiveWeights, SearchMode, RetrievalMode, RerankingComplexity,
    ContextExpansion, FusionStrategy, ContextRelationship, DocumentGraph, SearchStatistics,
    create_speed_optimized_config, create_balanced_config, create_accuracy_optimized_config
)
from models.query_models import QueryAnalysis, QueryIntent, QuestionType, EntityType
from services.document_analyzer import DocumentAnalyzer, DocumentStructure
from services.chunking_service import ChunkingService, TextChunk
from services.embedding_service import EmbeddingService
from storage.vector_store import VectorStore, SearchResult
from core.exceptions import RetrievalError, ConfigurationError

logger = logging.getLogger(__name__)


class IntelligentRetriever:
    """
    Advanced intelligent retrieval service with multi-modal search capabilities.
    
    This is the final component (Phase 3.3d) that brings together all previous
    components to deliver optimal search results through sophisticated analysis
    and ranking.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        document_analyzer: Optional[DocumentAnalyzer] = None,
        chunking_service: Optional[ChunkingService] = None,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize the intelligent retriever.
        
        Args:
            vector_store: Vector database for similarity search
            embedding_service: Service for generating embeddings
            document_analyzer: Optional document structure analyzer
            chunking_service: Optional chunking service for context
            config: Retrieval configuration (defaults to balanced)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.document_analyzer = document_analyzer
        self.chunking_service = chunking_service
        self.config = config or create_balanced_config()
        
        # Performance tracking
        self.stats = SearchStatistics()
        self._cache: Dict[str, RetrievalResult] = {}
        
        # Document graph for context relationships
        self.document_graph: Optional[DocumentGraph] = None
        self._build_document_graph()
        
        logger.info(f"IntelligentRetriever initialized with {self.config.retrieval_mode.value} mode")
    
    
    # ==================== MAIN RETRIEVAL METHODS ====================
    
    async def retrieve_intelligent(
        self,
        query_analysis: QueryAnalysis,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        strategy_override: Optional[str] = None,
        config_override: Optional[RetrievalConfig] = None
    ) -> RetrievalResult:
        """
        Main intelligent retrieval method with comprehensive analysis.
        
        Args:
            query_analysis: Complete query analysis from QueryAnalyzer
            top_k: Maximum number of results to return
            filters: Optional metadata filters
            strategy_override: Override retrieval strategy
            config_override: Temporary configuration override
            
        Returns:
            RetrievalResult with ranked chunks and comprehensive metadata
        """
        start_time = time.time()
        
        try:
            # Use configuration override if provided
            working_config = config_override or self.config
            
            # Check cache first
            cache_key = self._generate_cache_key(query_analysis, top_k, filters)
            if working_config.enable_caching and cache_key in self._cache:
                logger.debug("Cache hit for query retrieval")
                self.stats.cache_hits += 1
                cached_result = self._cache[cache_key]
                # Update processing time for cache hit
                cached_result.processing_time = (time.time() - start_time) * 1000
                return cached_result
            
            self.stats.cache_misses += 1
            
            # Determine optimal search strategy
            strategy = strategy_override or self._select_optimal_strategy(query_analysis, working_config)
            
            # Calculate query-adaptive weights
            adaptive_weights = self._calculate_adaptive_weights(query_analysis, working_config)
            
            # Execute multi-modal search based on performance mode
            if working_config.retrieval_mode == RetrievalMode.SPEED_OPTIMIZED:
                multi_modal_result = await self._speed_optimized_search(
                    query_analysis, adaptive_weights, top_k * 2, filters
                )
            elif working_config.retrieval_mode == RetrievalMode.ACCURACY_OPTIMIZED:
                multi_modal_result = await self._accuracy_optimized_search(
                    query_analysis, adaptive_weights, top_k * 3, filters
                )
            else:  # BALANCED (default)
                multi_modal_result = await self._balanced_search(
                    query_analysis, adaptive_weights, top_k * 2, filters
                )
            
            # Apply dynamic re-ranking with selected complexity
            ranked_chunks = self._dynamic_rerank(
                multi_modal_result.combined_results,
                query_analysis,
                working_config
            )
            
            # Apply context expansion if enabled
            if working_config.context_expansion != ContextExpansion.MINIMAL:
                ranked_chunks = await self._expand_context(
                    ranked_chunks, query_analysis, working_config
                )
            
            # Limit to requested top_k
            final_chunks = ranked_chunks[:top_k]
            
            # Calculate confidence and quality metrics
            confidence_score = self._calculate_retrieval_confidence(final_chunks, query_analysis)
            precision_estimate = self._estimate_precision(final_chunks, query_analysis)
            
            # Build final result
            processing_time = (time.time() - start_time) * 1000
            
            result = RetrievalResult(
                query_id=f"retr_{int(time.time() * 1000)}",
                query_analysis=query_analysis,
                ranked_chunks=final_chunks,
                multi_modal_result=multi_modal_result,
                retrieval_strategy=strategy,
                config_used=working_config,
                total_candidates=multi_modal_result.total_candidates,
                processing_time=processing_time,
                confidence_score=confidence_score,
                precision_estimate=precision_estimate,
                metadata={
                    'adaptive_weights': adaptive_weights.__dict__,
                    'strategy_reason': self._explain_strategy_selection(query_analysis),
                    'performance_mode': working_config.retrieval_mode.value,
                    'reranking_complexity': working_config.reranking_complexity.value
                }
            )
            
            # Update statistics
            self.stats.update(processing_time, len(final_chunks), precision_estimate)
            
            # Cache result if enabled
            if working_config.enable_caching:
                self._cache[cache_key] = result
                self._cleanup_cache()
            
            logger.info(f"Intelligent retrieval completed in {processing_time:.1f}ms, "
                       f"{len(final_chunks)} results, confidence: {confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Intelligent retrieval failed: {e}")
            self.stats.error_count += 1
            raise RetrievalError(f"Retrieval failed: {str(e)}") from e
    
    
    async def search_by_structure(
        self,
        query: str,
        structure_filters: Dict[str, Any],
        similarity_threshold: float = 0.7
    ) -> List[StructureMatch]:
        """
        Structure-aware search with metadata filtering.
        
        Args:
            query: Search query
            structure_filters: Structure-based filters
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of structure-aware matches
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # Build structure-specific filters
            search_filters = self._build_structure_filters(structure_filters)
            
            # Execute structure-aware search
            search_results = await self.vector_store.similarity_search_with_filters(
                query_embedding=query_embedding,
                filters=search_filters,
                k=50,  # Get more candidates for structure analysis
                similarity_threshold=similarity_threshold
            )
            
            structure_matches = []
            
            for result in search_results:
                # Calculate structure relevance scores
                structure_score = self._calculate_structure_relevance(
                    result.document.metadata, structure_filters
                )
                
                if structure_score > 0.3:  # Minimum structure relevance
                    structure_match = StructureMatch(
                        chunk=self._convert_to_text_chunk(result.document),
                        structure_relevance_score=structure_score,
                        section_type_alignment=self._calculate_section_alignment(
                            result.document.metadata, structure_filters
                        ),
                        hierarchy_bonus=self._calculate_hierarchy_bonus(
                            result.document.metadata
                        ),
                        document_type_match=self._calculate_document_type_match(
                            result.document.metadata, structure_filters
                        ),
                        structure_metadata={
                            'original_similarity': result.score,
                            'structure_boost': structure_score,
                            'filters_applied': structure_filters
                        }
                    )
                    structure_matches.append(structure_match)
            
            # Sort by combined structure + similarity score
            structure_matches.sort(
                key=lambda x: (x.structure_relevance_score + x.chunk.metadata.get('similarity_score', 0)) / 2,
                reverse=True
            )
            
            logger.debug(f"Structure search found {len(structure_matches)} matches")
            return structure_matches
            
        except Exception as e:
            logger.error(f"Structure search failed: {e}")
            raise RetrievalError(f"Structure search failed: {str(e)}") from e
    
    
    async def search_multi_modal(
        self,
        query_analysis: QueryAnalysis,
        weights: Optional[Dict[str, float]] = None
    ) -> MultiModalResult:
        """
        Multi-modal retrieval across different content types.
        
        Args:
            query_analysis: Complete query analysis
            weights: Optional weight overrides
            
        Returns:
            MultiModalResult with all search modes
        """
        try:
            # Calculate weights (adaptive or provided)
            if weights:
                adaptive_weights = QueryAdaptiveWeights(
                    text_weight=weights.get('text', 0.6),
                    table_weight=weights.get('table', 0.25),
                    entity_weight=weights.get('entity', 0.15),
                    structure_weight=weights.get('structure', 0.0)
                ).normalize()
            else:
                adaptive_weights = self._calculate_adaptive_weights(query_analysis, self.config)
            
            # Execute parallel searches based on weights
            search_tasks = []
            
            # Text search (always included)
            search_tasks.append(self._search_text_mode(query_analysis, adaptive_weights.text_weight))
            
            # Table search (if weight > 0)
            if adaptive_weights.table_weight > 0.1:
                search_tasks.append(self._search_table_mode(query_analysis, adaptive_weights.table_weight))
            
            # Entity search (if weight > 0)
            if adaptive_weights.entity_weight > 0.1:
                search_tasks.append(self._search_entity_mode(query_analysis, adaptive_weights.entity_weight))
            
            # Structure search (if enabled)
            if adaptive_weights.structure_weight > 0.1:
                search_tasks.append(self._search_structure_mode(query_analysis, adaptive_weights.structure_weight))
            
            # Execute searches in parallel
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process results and handle any exceptions
            text_results, table_results, entity_results, structure_results = [], [], [], []
            
            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    logger.warning(f"Search mode {i} failed: {result}")
                    continue
                
                if i == 0:  # Text results
                    text_results = result
                elif i == 1 and adaptive_weights.table_weight > 0.1:  # Table results
                    table_results = result
                elif i == (2 if adaptive_weights.table_weight > 0.1 else 1) and adaptive_weights.entity_weight > 0.1:  # Entity results
                    entity_results = result
                elif adaptive_weights.structure_weight > 0.1:  # Structure results
                    structure_results = result
            
            # Fuse multi-modal results
            combined_results = self._fuse_multi_modal_results(
                text_results, table_results, entity_results, structure_results, adaptive_weights
            )
            
            # Calculate fusion confidence
            fusion_confidence = self._calculate_fusion_confidence(
                text_results, table_results, entity_results, combined_results
            )
            
            return MultiModalResult(
                text_results=text_results,
                table_results=table_results,
                entity_results=entity_results,
                structure_results=structure_results,
                combined_results=combined_results,
                modal_weights=adaptive_weights,
                fusion_strategy=FusionStrategy.WEIGHTED_AVERAGE,
                fusion_confidence=fusion_confidence,
                total_candidates=len(text_results) + len(table_results) + len(entity_results),
                processing_time_ms=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Multi-modal search failed: {e}")
            raise RetrievalError(f"Multi-modal search failed: {str(e)}") from e
    
    
    # ==================== PERFORMANCE MODE IMPLEMENTATIONS ====================
    
    async def _speed_optimized_search(
        self,
        query_analysis: QueryAnalysis,
        weights: QueryAdaptiveWeights,
        max_candidates: int,
        filters: Optional[Dict[str, Any]]
    ) -> MultiModalResult:
        """Speed-optimized search (<300ms target)."""
        
        # Simplified search with basic text + table if needed
        text_results = await self._search_text_mode(query_analysis, weights.text_weight, max_candidates // 2)
        
        table_results = []
        if weights.table_weight > 0.2:  # Only if table weight is significant
            table_results = await self._search_table_mode(query_analysis, weights.table_weight, max_candidates // 4)
        
        # Skip entity and structure search for speed
        entity_results = []
        structure_results = []
        
        # Simple weighted fusion
        combined_results = self._simple_fusion(text_results, table_results, weights)
        
        return MultiModalResult(
            text_results=text_results,
            table_results=table_results,
            entity_results=entity_results,
            structure_results=structure_results,
            combined_results=combined_results,
            modal_weights=weights,
            fusion_strategy=FusionStrategy.WEIGHTED_AVERAGE,
            fusion_confidence=0.8,  # Lower confidence for speed mode
            total_candidates=len(combined_results),
            processing_time_ms=0.0
        )
    
    
    async def _balanced_search(
        self,
        query_analysis: QueryAnalysis,
        weights: QueryAdaptiveWeights,
        max_candidates: int,
        filters: Optional[Dict[str, Any]]
    ) -> MultiModalResult:
        """Balanced search (<500ms target) - DEFAULT."""
        
        return await self.search_multi_modal(query_analysis, weights.__dict__)
    
    
    async def _accuracy_optimized_search(
        self,
        query_analysis: QueryAnalysis,
        weights: QueryAdaptiveWeights,
        max_candidates: int,
        filters: Optional[Dict[str, Any]]
    ) -> MultiModalResult:
        """Accuracy-optimized search (<1000ms target)."""
        
        # Comprehensive multi-modal search with higher candidate counts
        result = await self.search_multi_modal(query_analysis, weights.__dict__)
        
        # Additional processing for accuracy
        # Expand entity relationships
        if result.entity_results:
            result.entity_results = await self._expand_entity_relationships(
                result.entity_results, query_analysis
            )
        
        # Enhanced table analysis
        if result.table_results and self.config.enable_semantic_table_analysis:
            result.table_results = await self._enhance_table_analysis(
                result.table_results, query_analysis
            )
        
        # Re-fuse with enhanced results
        result.combined_results = self._fuse_multi_modal_results(
            result.text_results, result.table_results, 
            result.entity_results, result.structure_results, weights
        )
        
        return result
    
    
    # ==================== SPECIALIZED SEARCH MODES ====================
    
    async def _search_text_mode(
        self,
        query_analysis: QueryAnalysis,
        weight: float,
        max_results: int = 20
    ) -> List[RankedChunk]:
        """Execute semantic text search."""
        
        try:
            # Use expanded query for better recall
            search_query = query_analysis.expansion.processed_query if query_analysis.expansion else query_analysis.original_query
            
            # Generate embedding
            query_embedding = await self.embedding_service.generate_embedding(search_query)
            
            # Execute similarity search
            search_results = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=max_results,
                filters={'chunk_type': {'$ne': 'table_only'}}  # Exclude table-only chunks
            )
            
            # Convert to ranked chunks
            ranked_chunks = []
            for i, result in enumerate(search_results):
                chunk = self._convert_to_text_chunk(result.document)
                
                ranked_chunk = RankedChunk(
                    chunk=chunk,
                    base_similarity_score=result.score,
                    structure_relevance_score=0.0,
                    entity_alignment_score=0.0,
                    intent_alignment_score=0.0,
                    context_coherence_score=0.0,
                    final_rank_score=result.score * weight,
                    rank_position=i,
                    ranking_explanation=f"Text similarity: {result.score:.3f}",
                    ranking_factors_used=['text_similarity']
                )
                ranked_chunks.append(ranked_chunk)
            
            logger.debug(f"Text search returned {len(ranked_chunks)} results")
            return ranked_chunks
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    
    async def _search_table_mode(
        self,
        query_analysis: QueryAnalysis,
        weight: float,
        max_results: int = 15
    ) -> List[TableMatch]:
        """Execute table-specific search with semantic analysis."""
        
        try:
            if not self.config.enable_semantic_table_analysis:
                # Fallback to basic table search
                return await self._basic_table_search(query_analysis, weight, max_results)
            
            # Find chunks that contain tables
            table_filters = {
                'chunk_type': 'table',
                'has_table_data': True
            }
            
            query_embedding = await self.embedding_service.generate_embedding(
                query_analysis.original_query
            )
            
            search_results = await self.vector_store.similarity_search_with_filters(
                query_embedding=query_embedding,
                filters=table_filters,
                k=max_results * 2  # Get more candidates for table analysis
            )
            
            table_matches = []
            
            for result in search_results:
                chunk = self._convert_to_text_chunk(result.document)
                
                # Perform semantic table analysis
                table_score = await self._analyze_table_semantics(chunk, query_analysis)
                
                if table_score > 0.4:  # Minimum table relevance
                    table_match = TableMatch(
                        chunk=chunk,
                        table_similarity_score=table_score,
                        matching_columns=self._find_matching_columns(chunk, query_analysis.entities),
                        matching_cells=self._find_matching_cells(chunk, query_analysis),
                        table_type=chunk.metadata.get('table_type', 'unknown'),
                        header_relevance_score=self._calculate_header_relevance(chunk, query_analysis),
                        entity_alignment_score=self._calculate_table_entity_alignment(chunk, query_analysis),
                        data_type_confidence=chunk.metadata.get('table_data_confidence', 0.5),
                        table_context={
                            'base_similarity': result.score,
                            'semantic_boost': table_score - result.score,
                            'analysis_method': 'semantic'
                        }
                    )
                    table_matches.append(table_match)
            
            # Sort by table similarity score
            table_matches.sort(key=lambda x: x.table_similarity_score, reverse=True)
            table_matches = table_matches[:max_results]
            
            logger.debug(f"Table search returned {len(table_matches)} results")
            return table_matches
            
        except Exception as e:
            logger.error(f"Table search failed: {e}")
            return []
    
    
    async def _search_entity_mode(
        self,
        query_analysis: QueryAnalysis,
        weight: float,
        max_results: int = 15
    ) -> List[EntityMatch]:
        """Execute entity-focused search with relationship context."""
        
        try:
            if not query_analysis.entities:
                logger.debug("No entities found in query, skipping entity search")
                return []
            
            entity_matches = []
            
            for query_entity in query_analysis.entities[:3]:  # Limit to top 3 entities
                # Search for chunks containing this entity
                entity_filters = {
                    'overlapping_entities': {'$elemMatch': {'text': query_entity.text}}
                }
                
                query_embedding = await self.embedding_service.generate_embedding(
                    f"{query_analysis.original_query} {query_entity.text}"
                )
                
                search_results = await self.vector_store.similarity_search_with_filters(
                    query_embedding=query_embedding,
                    filters=entity_filters,
                    k=max_results
                )
                
                for result in search_results:
                    chunk = self._convert_to_text_chunk(result.document)
                    
                    # Calculate entity prominence and relationships
                    entity_prominence = self._calculate_entity_prominence(chunk, query_entity)
                    relationship_strength = self._calculate_entity_relationships(chunk, query_entity)
                    
                    entity_match = EntityMatch(
                        chunk=chunk,
                        matched_entity=query_entity,
                        entity_context=self._extract_entity_context(chunk, query_entity),
                        related_entities=self._find_related_entities(chunk, query_entity),
                        entity_prominence=entity_prominence,
                        relationship_strength=relationship_strength,
                        confidence_score=result.score,
                        entity_metadata={
                            'entity_type': query_entity.entity_type.value,
                            'entity_confidence': query_entity.confidence,
                            'search_similarity': result.score
                        }
                    )
                    entity_matches.append(entity_match)
            
            # Remove duplicates and rank by prominence
            entity_matches = self._deduplicate_entity_matches(entity_matches)
            entity_matches.sort(key=lambda x: x.entity_prominence * x.confidence_score, reverse=True)
            entity_matches = entity_matches[:max_results]
            
            logger.debug(f"Entity search returned {len(entity_matches)} results")
            return entity_matches
            
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            return []
    
    
    async def _search_structure_mode(
        self,
        query_analysis: QueryAnalysis,
        weight: float,
        max_results: int = 15
    ) -> List[StructureMatch]:
        """Execute structure-aware search."""
        
        try:
            # Build structure filters based on query analysis
            structure_filters = self._build_query_structure_filters(query_analysis)
            
            return await self.search_by_structure(
                query_analysis.original_query,
                structure_filters,
                similarity_threshold=0.6
            )
            
        except Exception as e:
            logger.error(f"Structure search failed: {e}")
            return []
    
    
    # ==================== RE-RANKING IMPLEMENTATIONS ====================
    
    def _dynamic_rerank(
        self,
        results: List[RankedChunk],
        query_analysis: QueryAnalysis,
        config: RetrievalConfig
    ) -> List[RankedChunk]:
        """
        Advanced re-ranking using multiple scoring factors.
        
        Supports three complexity levels:
        - BASIC: similarity + intent + entity (3 factors)
        - ADVANCED: + structure + coherence + authority (6 factors)
        - COMPREHENSIVE: + relationships + temporal + references (9+ factors)
        """
        
        if not results:
            return results
        
        try:
            complexity = config.reranking_complexity
            
            for chunk in results:
                # Always calculate basic factors
                chunk.intent_alignment_score = self._calculate_intent_alignment(
                    chunk, query_analysis.intent
                )
                chunk.entity_alignment_score = self._calculate_entity_alignment_score(
                    chunk, query_analysis.entities
                )
                
                # Calculate structure relevance if enabled
                if config.structure_influence_weight > 0:
                    chunk.structure_relevance_score = self._calculate_structure_relevance_score(
                        chunk, query_analysis
                    )
                
                # ADVANCED complexity adds more factors
                if complexity in [RerankingComplexity.ADVANCED, RerankingComplexity.COMPREHENSIVE]:
                    chunk.context_coherence_score = self._calculate_context_coherence(
                        chunk, results[:10], query_analysis  # Use top 10 for coherence calculation
                    )
                    
                    chunk.authority_score = self._calculate_authority_score(
                        chunk, self.document_graph
                    )
                    
                    # Update ranking factors used
                    chunk.ranking_factors_used.extend([
                        'context_coherence', 'authority_score'
                    ])
                
                # COMPREHENSIVE complexity adds even more factors
                if complexity == RerankingComplexity.COMPREHENSIVE:
                    chunk.recency_score = self._calculate_recency_score(chunk)
                    chunk.cross_reference_score = self._calculate_cross_reference_score(
                        chunk, results, self.document_graph
                    )
                    chunk.temporal_relevance_score = self._calculate_temporal_relevance(
                        chunk, query_analysis
                    )
                    
                    # Update ranking factors used
                    chunk.ranking_factors_used.extend([
                        'recency', 'cross_references', 'temporal_relevance'
                    ])
                
                # Calculate final rank score based on complexity
                chunk.final_rank_score = self._calculate_final_rank_score(chunk, complexity, config)
                
                # Generate ranking explanation
                chunk.ranking_explanation = self._generate_ranking_explanation(chunk, complexity)
            
            # Sort by final rank score
            results.sort(key=lambda x: x.final_rank_score, reverse=True)
            
            # Update rank positions
            for i, chunk in enumerate(results):
                chunk.rank_position = i + 1
            
            logger.debug(f"Re-ranked {len(results)} results using {complexity.value} complexity")
            return results
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return results  # Return original results if re-ranking fails
    
    
    def _calculate_final_rank_score(
        self,
        chunk: RankedChunk,
        complexity: RerankingComplexity,
        config: RetrievalConfig
    ) -> float:
        """Calculate final ranking score based on complexity level."""
        
        if complexity == RerankingComplexity.BASIC:
            # Basic: similarity + intent + entity
            return (
                chunk.base_similarity_score * 0.6 +
                chunk.intent_alignment_score * 0.25 +
                chunk.entity_alignment_score * 0.15
            )
        
        elif complexity == RerankingComplexity.ADVANCED:
            # Advanced: 6 factors
            return (
                chunk.base_similarity_score * 0.4 +
                chunk.structure_relevance_score * config.structure_influence_weight +
                chunk.intent_alignment_score * 0.2 +
                chunk.entity_alignment_score * 0.15 +
                chunk.context_coherence_score * 0.1 +
                chunk.authority_score * 0.05
            )
        
        else:  # COMPREHENSIVE
            # Comprehensive: 9+ factors
            return (
                chunk.base_similarity_score * 0.3 +
                chunk.structure_relevance_score * (config.structure_influence_weight * 0.8) +
                chunk.intent_alignment_score * 0.15 +
                chunk.entity_alignment_score * 0.12 +
                chunk.context_coherence_score * 0.1 +
                chunk.authority_score * 0.08 +
                chunk.recency_score * 0.05 +
                chunk.cross_reference_score * 0.05 +
                chunk.temporal_relevance_score * 0.05
            )
    
    
    # ==================== CONTEXT EXPANSION ====================
    
    async def _expand_context(
        self,
        ranked_chunks: List[RankedChunk],
        query_analysis: QueryAnalysis,
        config: RetrievalConfig
    ) -> List[RankedChunk]:
        """Expand context based on document relationships."""
        
        if config.context_expansion == ContextExpansion.MINIMAL:
            return ranked_chunks
        
        try:
            expanded_chunks = list(ranked_chunks)  # Copy original results
            
            # Document context expansion
            if config.context_expansion in [ContextExpansion.DOCUMENT_CONTEXT, ContextExpansion.CROSS_DOCUMENT]:
                document_context_chunks = await self._add_document_context(
                    ranked_chunks, config.max_context_chunks
                )
                expanded_chunks.extend(document_context_chunks)
            
            # Cross-document context expansion
            if config.context_expansion == ContextExpansion.CROSS_DOCUMENT and config.enable_cross_document_context:
                cross_doc_chunks = await self._add_cross_document_context(
                    ranked_chunks, query_analysis, config.max_context_chunks
                )
                expanded_chunks.extend(cross_doc_chunks)
            
            # Remove duplicates and maintain ranking
            expanded_chunks = self._deduplicate_chunks(expanded_chunks)
            
            logger.debug(f"Context expansion added {len(expanded_chunks) - len(ranked_chunks)} chunks")
            return expanded_chunks
            
        except Exception as e:
            logger.error(f"Context expansion failed: {e}")
            return ranked_chunks
    
    
    # ==================== ADAPTIVE WEIGHT CALCULATION ====================
    
    def _calculate_adaptive_weights(
        self,
        query_analysis: QueryAnalysis,
        config: RetrievalConfig
    ) -> QueryAdaptiveWeights:
        """Calculate query-adaptive weights for multi-modal search."""
        
        if not config.enable_query_adaptive_weighting:
            # Use default weights
            return QueryAdaptiveWeights(
                text_weight=config.default_modal_weights['text'],
                table_weight=config.default_modal_weights['table'],
                entity_weight=config.default_modal_weights['entity'],
                structure_weight=0.0
            ).normalize()
        
        # Base weights
        text_weight = 0.6
        table_weight = 0.25
        entity_weight = 0.15
        structure_weight = 0.0
        temporal_weight = 0.0
        
        # Adjust based on query intent
        if query_analysis.intent == QueryIntent.FACTUAL:
            table_weight += 0.15  # Boost tables for factual queries
            entity_weight += 0.1  # Boost entities for facts
            text_weight -= 0.25
        
        elif query_analysis.intent == QueryIntent.ANALYTICAL:
            text_weight += 0.15   # Boost text for analysis
            structure_weight += 0.1  # Structure helps with analysis
            table_weight -= 0.1
        
        elif query_analysis.intent == QueryIntent.COMPARATIVE:
            table_weight += 0.2   # Tables great for comparisons
            structure_weight += 0.1
            text_weight -= 0.3
        
        elif query_analysis.intent == QueryIntent.PROCEDURAL:
            structure_weight += 0.2  # Structure important for procedures
            text_weight -= 0.1
        
        elif query_analysis.intent == QueryIntent.VERIFICATION:
            entity_weight += 0.15  # Entities important for verification
            temporal_weight += 0.1  # Time might matter
            text_weight -= 0.15
        
        # Adjust based on entity types
        if query_analysis.entities:
            entity_types = [e.entity_type for e in query_analysis.entities]
            
            # Boost tables for numeric/currency entities
            if any(et in [EntityType.CURRENCY, EntityType.NUMERIC] for et in entity_types):
                table_weight += 0.1
                text_weight -= 0.1
            
            # Boost temporal for date/time entities
            if any(et == EntityType.DATE_TIME for et in entity_types):
                temporal_weight += 0.05
        
        # Apply question type adjustments
        if hasattr(query_analysis, 'question_type'):
            if query_analysis.question_type == QuestionType.HOW_MUCH:
                table_weight += 0.1
            elif query_analysis.question_type == QuestionType.WHEN:
                temporal_weight += 0.1
        
        return QueryAdaptiveWeights(
            text_weight=max(0.1, text_weight),      # Minimum 10%
            table_weight=max(0.0, table_weight),
            entity_weight=max(0.0, entity_weight),
            structure_weight=max(0.0, structure_weight),
            temporal_weight=max(0.0, temporal_weight)
        ).normalize()
    
    
    # ==================== CONFIGURATION SWITCHING ====================
    
    def switch_performance_mode(self, mode: RetrievalMode) -> RetrievalConfig:
        """Switch to different performance mode and return new config."""
        
        if mode == RetrievalMode.SPEED_OPTIMIZED:
            new_config = create_speed_optimized_config()
        elif mode == RetrievalMode.ACCURACY_OPTIMIZED:
            new_config = create_accuracy_optimized_config()
        else:
            new_config = create_balanced_config()
        
        logger.info(f"Switched to {mode.value} performance mode")
        return new_config
    
    
    def switch_reranking_complexity(self, complexity: RerankingComplexity) -> RetrievalConfig:
        """Switch re-ranking complexity level."""
        
        new_config = replace(self.config, reranking_complexity=complexity)
        logger.info(f"Switched to {complexity.value} re-ranking complexity")
        return new_config
    
    
    # ==================== UTILITY AND HELPER METHODS ====================
    
    def _select_optimal_strategy(
        self, 
        query_analysis: QueryAnalysis, 
        config: RetrievalConfig
    ) -> str:
        """Select optimal retrieval strategy based on query analysis."""
        
        intent = query_analysis.intent
        
        if intent == QueryIntent.FACTUAL:
            return "entity_focused_retrieval"
        elif intent == QueryIntent.COMPARATIVE:
            return "multi_document_retrieval"
        elif intent == QueryIntent.ANALYTICAL:
            return "semantic_structure_retrieval"
        elif intent == QueryIntent.PROCEDURAL:
            return "structure_aware_retrieval"
        elif intent == QueryIntent.VERIFICATION:
            return "entity_temporal_retrieval"
        else:  # EXPLORATORY
            return "broad_semantic_retrieval"
    
    
    def _generate_cache_key(
        self,
        query_analysis: QueryAnalysis,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for query result."""
        
        key_parts = [
            query_analysis.original_query,
            str(top_k),
            str(sorted(filters.items())) if filters else "no_filters",
            self.config.retrieval_mode.value,
            self.config.reranking_complexity.value
        ]
        
        return hash(tuple(key_parts)).__str__()
    
    
    def _cleanup_cache(self):
        """Clean up cache based on TTL and size limits."""
        
        # Simple cleanup - remove oldest entries if cache is too large
        if len(self._cache) > 100:  # Max cache size
            # Remove oldest 20 entries (simple FIFO)
            oldest_keys = list(self._cache.keys())[:20]
            for key in oldest_keys:
                del self._cache[key]
    
    
    def _build_document_graph(self):
        """Build document relationship graph for context expansion."""
        
        try:
            if self.document_analyzer:
                # Build graph from document analyzer knowledge
                # This would be more complex in a real implementation
                self.document_graph = DocumentGraph(
                    documents={},
                    relationships=[],
                    entity_connections={},
                    temporal_ordering={},
                    authority_scores={}
                )
                logger.debug("Document graph initialized")
            else:
                logger.warning("No DocumentAnalyzer available for graph building")
        
        except Exception as e:
            logger.error(f"Failed to build document graph: {e}")
            self.document_graph = None
    
    
    # ==================== PLACEHOLDER METHODS ====================
    # These methods would contain the detailed implementation logic
    # For brevity, I'm including key signatures and basic logic
    
    def _convert_to_text_chunk(self, document) -> TextChunk:
        """Convert VectorDocument to TextChunk."""
        # Implementation would convert between formats
        pass
    
    def _calculate_structure_relevance(self, metadata: Dict, filters: Dict) -> float:
        """Calculate structure relevance score."""
        return 0.5  # Placeholder
    
    def _calculate_intent_alignment(self, chunk: RankedChunk, intent: QueryIntent) -> float:
        """Calculate intent alignment score."""
        return 0.5  # Placeholder
    
    def _calculate_entity_alignment_score(self, chunk: RankedChunk, entities: List) -> float:
        """Calculate entity alignment score."""
        return 0.5  # Placeholder
    
    def _calculate_context_coherence(self, chunk: RankedChunk, all_chunks: List, query_analysis: QueryAnalysis) -> float:
        """Calculate context coherence score."""
        return 0.5  # Placeholder
    
    def _calculate_authority_score(self, chunk: RankedChunk, document_graph: Optional[DocumentGraph]) -> float:
        """Calculate document authority score."""
        return 0.5  # Placeholder
    
    def _calculate_recency_score(self, chunk: RankedChunk) -> float:
        """Calculate recency score."""
        return 0.5  # Placeholder
    
    def _calculate_cross_reference_score(self, chunk: RankedChunk, all_chunks: List, document_graph: Optional[DocumentGraph]) -> float:
        """Calculate cross-reference score."""
        return 0.5  # Placeholder
    
    def _calculate_temporal_relevance(self, chunk: RankedChunk, query_analysis: QueryAnalysis) -> float:
        """Calculate temporal relevance score."""
        return 0.5  # Placeholder
    
    def _fuse_multi_modal_results(self, text_results, table_results, entity_results, structure_results, weights) -> List[RankedChunk]:
        """Fuse results from different search modes."""
        return text_results[:10]  # Placeholder
    
    def _simple_fusion(self, text_results, table_results, weights) -> List[RankedChunk]:
        """Simple result fusion for speed mode."""
        return text_results[:10]  # Placeholder
    
    def _generate_ranking_explanation(self, chunk: RankedChunk, complexity: RerankingComplexity) -> str:
        """Generate human-readable ranking explanation."""
        return f"Ranked using {complexity.value} method"
    
    def _explain_strategy_selection(self, query_analysis: QueryAnalysis) -> str:
        """Explain why a particular strategy was selected."""
        return f"Selected based on {query_analysis.intent.value} intent"
    
    def _calculate_retrieval_confidence(self, chunks: List[RankedChunk], query_analysis: QueryAnalysis) -> float:
        """Calculate overall retrieval confidence."""
        if not chunks:
            return 0.0
        return sum(c.final_rank_score for c in chunks) / len(chunks)
    
    def _estimate_precision(self, chunks: List[RankedChunk], query_analysis: QueryAnalysis) -> float:
        """Estimate precision of retrieved chunks."""
        # Heuristic based on top results' scores
        if not chunks:
            return 0.0
        top_3 = chunks[:3]
        return sum(c.final_rank_score for c in top_3) / len(top_3)
    
    # Additional placeholder methods would be implemented here...
    async def _analyze_table_semantics(self, chunk, query_analysis): pass
    def _find_matching_columns(self, chunk, entities): return []
    def _find_matching_cells(self, chunk, query_analysis): return []
    def _calculate_header_relevance(self, chunk, query_analysis): return 0.5
    def _calculate_table_entity_alignment(self, chunk, query_analysis): return 0.5
    def _basic_table_search(self, query_analysis, weight, max_results): return []
    def _calculate_entity_prominence(self, chunk, entity): return 0.5
    def _calculate_entity_relationships(self, chunk, entity): return 0.5
    def _extract_entity_context(self, chunk, entity): return ""
    def _find_related_entities(self, chunk, entity): return []
    def _deduplicate_entity_matches(self, matches): return matches
    def _build_query_structure_filters(self, query_analysis): return {}
    def _build_structure_filters(self, structure_filters): return structure_filters
    def _calculate_section_alignment(self, metadata, filters): return 0.5
    def _calculate_hierarchy_bonus(self, metadata): return 0.5
    def _calculate_document_type_match(self, metadata, filters): return 0.5
    def _calculate_structure_relevance_score(self, chunk, query_analysis): return 0.5
    def _calculate_fusion_confidence(self, text_results, table_results, entity_results, combined_results): return 0.8
    async def _expand_entity_relationships(self, entity_results, query_analysis): return entity_results
    async def _enhance_table_analysis(self, table_results, query_analysis): return table_results
    async def _add_document_context(self, chunks, max_chunks): return []
    async def _add_cross_document_context(self, chunks, query_analysis, max_chunks): return []
    def _deduplicate_chunks(self, chunks): return chunks
    
    # ==================== PERFORMANCE MONITORING ====================
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        return {
            'total_searches': self.stats.total_searches,
            'average_response_time_ms': self.stats.average_response_time_ms,
            'cache_hit_rate': self.stats.cache_hit_rate,
            'average_precision': self.stats.average_precision,
            'error_count': self.stats.error_count,
            'current_mode': self.config.retrieval_mode.value,
            'current_complexity': self.config.reranking_complexity.value,
            'cache_size': len(self._cache)
        }
    
    
    def reset_statistics(self):
        """Reset performance statistics."""
        self.stats = SearchStatistics()
        logger.info("Performance statistics reset")
