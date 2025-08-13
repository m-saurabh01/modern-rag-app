# Phase 3.3d: Intelligent Retrieval - Implementation Plan

## Overview
Phase 3.3d is the final component of our advanced RAG system, implementing intelligent retrieval that leverages document structure analysis and query understanding to deliver optimal search results with context-aware ranking and multi-modal retrieval capabilities.

## Objectives

### Primary Goals
- **Structure-Aware Similarity Search**: Use document structure metadata for enhanced retrieval
- **Multi-Modal Retrieval**: Combine text, tables, and entities in unified search
- **Dynamic Re-Ranking**: Adjust results based on query analysis and document structure
- **Context-Enhanced Scoring**: Incorporate document relationships and hierarchies
- **Adaptive Retrieval Strategies**: Select optimal search approaches based on query characteristics

### Integration Points
- **QueryAnalyzer Integration**: Use query analysis to guide retrieval strategy
- **DocumentAnalyzer Integration**: Leverage document structure for better relevance
- **ChunkingService Integration**: Optimize chunk selection based on structure metadata
- **VectorStore Integration**: Enhanced similarity search with metadata filtering

## Core Components to Build

### 1. **IntelligentRetriever Service** (`services/intelligent_retriever.py`)

#### Primary Responsibilities
- **Multi-Modal Search**: Search across text content, table data, and extracted entities
- **Structure-Aware Ranking**: Use document structure to enhance relevance scoring
- **Query-Adaptive Retrieval**: Select retrieval strategies based on query analysis
- **Context Preservation**: Maintain document relationships and hierarchies in results
- **Result Optimization**: Re-rank and filter results for optimal RAG performance

#### Key Methods to Implement

##### `retrieve_intelligent(query_analysis: QueryAnalysis, top_k: int = 10) -> RetrievalResult`
**Purpose**: Main intelligent retrieval method
```python
async def retrieve_intelligent(
    self,
    query_analysis: QueryAnalysis,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    strategy_override: Optional[RetrievalStrategy] = None
) -> RetrievalResult:
    """
    Intelligent retrieval using query analysis and document structure.
    
    Args:
        query_analysis: Complete query analysis from QueryAnalyzer
        top_k: Maximum number of results to return
        filters: Optional metadata filters
        strategy_override: Optional strategy override
        
    Returns:
        RetrievalResult with ranked chunks and metadata
    """
```

##### `search_by_structure(query: str, structure_filters: Dict) -> List[StructuredChunk]`
**Purpose**: Search using document structure information
```python
async def search_by_structure(
    self,
    query: str,
    structure_filters: Dict[str, Any],
    similarity_threshold: float = 0.7
) -> List[StructuredChunk]:
    """
    Structure-aware search with metadata filtering.
    
    Structure Filters:
    - document_type: ['government', 'technical', 'business']
    - section_type: ['header', 'paragraph', 'table', 'list_item']
    - section_level: [0, 1, 2, 3, 4]
    - contains_tables: True/False
    - entity_types: ['PERSON', 'ORGANIZATION', 'DATE']
    - complexity_score: [min, max]
    """
```

##### `search_multi_modal(query_analysis: QueryAnalysis) -> MultiModalResult`
**Purpose**: Search across text, tables, and entities simultaneously
```python
async def search_multi_modal(
    self,
    query_analysis: QueryAnalysis,
    weights: Optional[Dict[str, float]] = None
) -> MultiModalResult:
    """
    Multi-modal retrieval across different content types.
    
    Search Modes:
    - TEXT: Standard semantic similarity search
    - TABLE: Table-specific search with column/row matching
    - ENTITY: Entity-focused search with relationship context
    - HYBRID: Combined search with weighted scoring
    
    Default weights: {'text': 0.6, 'table': 0.25, 'entity': 0.15}
    """
```

##### `dynamic_rerank(results: List[Chunk], query_analysis: QueryAnalysis) -> List[RankedChunk]`
**Purpose**: Re-rank results using advanced scoring
```python
def dynamic_rerank(
    self,
    results: List[Chunk],
    query_analysis: QueryAnalysis,
    context_chunks: Optional[List[Chunk]] = None
) -> List[RankedChunk]:
    """
    Advanced re-ranking using multiple scoring factors.
    
    Ranking Factors:
    - Semantic similarity (base score)
    - Query intent alignment
    - Entity overlap and relevance
    - Document structure relevance
    - Context coherence
    - Recency and authority (if available)
    """
```

### 2. **Data Models** (`models/retrieval_models.py`)

#### Core Data Structures

##### `RetrievalResult`
```python
@dataclass
class RetrievalResult:
    """Complete retrieval result with metadata."""
    query_id: str
    query_analysis: QueryAnalysis
    ranked_chunks: List[RankedChunk]
    retrieval_strategy: RetrievalStrategy
    total_candidates: int
    processing_time: float
    confidence_score: float
    metadata: Dict[str, Any]
```

##### `RankedChunk`
```python
@dataclass
class RankedChunk:
    """Chunk with ranking information."""
    chunk: TextChunk
    base_similarity_score: float
    structure_relevance_score: float
    entity_alignment_score: float
    intent_alignment_score: float
    context_coherence_score: float
    final_rank_score: float
    rank_position: int
    ranking_explanation: str
```

##### `MultiModalResult`
```python
@dataclass
class MultiModalResult:
    """Multi-modal search result."""
    text_results: List[RankedChunk]
    table_results: List[TableMatch]
    entity_results: List[EntityMatch]
    combined_results: List[RankedChunk]
    modal_weights: Dict[str, float]
    fusion_strategy: str
```

##### `RetrievalStrategy`
```python
@dataclass
class RetrievalStrategy:
    """Retrieval strategy configuration."""
    name: str
    search_modes: List[SearchMode]
    filters: Dict[str, Any]
    ranking_weights: Dict[str, float]
    similarity_threshold: float
    max_candidates: int
    reranking_enabled: bool
    context_expansion: bool
```

### 3. **Advanced Retrieval Strategies**

#### Strategy 1: Intent-Aware Retrieval
```python
class IntentAwareRetrieval:
    """Retrieval strategy based on query intent."""
    
    def configure_for_intent(self, intent: QueryIntent) -> RetrievalStrategy:
        """Configure retrieval strategy based on query intent."""
        
        strategies = {
            QueryIntent.FACTUAL: RetrievalStrategy(
                name="factual_retrieval",
                search_modes=[SearchMode.TEXT, SearchMode.TABLE, SearchMode.ENTITY],
                filters={
                    'prefer_tables': True,
                    'section_types': ['paragraph', 'table', 'list_item'],
                    'entity_rich': True
                },
                ranking_weights={
                    'similarity': 0.4,
                    'entity_match': 0.3,
                    'structure_relevance': 0.2,
                    'table_presence': 0.1
                },
                similarity_threshold=0.75,
                max_candidates=50,
                reranking_enabled=True
            ),
            
            QueryIntent.ANALYTICAL: RetrievalStrategy(
                name="analytical_retrieval",
                search_modes=[SearchMode.TEXT, SearchMode.CONTEXT],
                filters={
                    'section_types': ['header', 'paragraph'],
                    'min_section_level': 0,
                    'include_context': True
                },
                ranking_weights={
                    'similarity': 0.5,
                    'context_coherence': 0.3,
                    'structure_relevance': 0.2
                },
                similarity_threshold=0.65,
                max_candidates=30,
                context_expansion=True
            ),
            
            QueryIntent.PROCEDURAL: RetrievalStrategy(
                name="procedural_retrieval",
                search_modes=[SearchMode.TEXT, SearchMode.SEQUENCE],
                filters={
                    'section_types': ['list_item', 'paragraph'],
                    'has_sequence': True,
                    'document_types': ['manual', 'guide', 'procedure']
                },
                ranking_weights={
                    'similarity': 0.4,
                    'sequence_relevance': 0.3,
                    'structure_relevance': 0.3
                },
                similarity_threshold=0.7,
                max_candidates=25
            )
        }
        
        return strategies.get(intent, self._default_strategy())
```

#### Strategy 2: Document-Type-Aware Retrieval
```python
class DocumentTypeRetrieval:
    """Retrieval optimized for specific document types."""
    
    def configure_for_document_type(self, doc_type: str) -> RetrievalStrategy:
        """Configure retrieval for specific document types."""
        
        if doc_type == 'government':
            return RetrievalStrategy(
                name="government_doc_retrieval",
                filters={
                    'document_type': 'government',
                    'entity_types': ['DEPARTMENT', 'REFERENCE_NUMBER', 'OFFICIAL_TITLE'],
                    'section_types': ['header', 'paragraph', 'table']
                },
                ranking_weights={
                    'similarity': 0.4,
                    'entity_match': 0.3,
                    'authority_score': 0.2,
                    'recency': 0.1
                }
            )
        
        elif doc_type == 'technical':
            return RetrievalStrategy(
                name="technical_doc_retrieval",
                filters={
                    'document_type': 'technical',
                    'entity_types': ['VERSION_NUMBER', 'REQUIREMENT_ID'],
                    'has_tables': True
                },
                ranking_weights={
                    'similarity': 0.5,
                    'technical_relevance': 0.3,
                    'table_presence': 0.2
                }
            )
        
        # ... more document type strategies
```

#### Strategy 3: Multi-Modal Fusion
```python
class MultiModalFusion:
    """Combine results from different search modes."""
    
    async def fuse_results(
        self,
        text_results: List[RankedChunk],
        table_results: List[TableMatch], 
        entity_results: List[EntityMatch],
        weights: Dict[str, float]
    ) -> List[RankedChunk]:
        """Fuse multi-modal search results."""
        
        # Convert all results to unified format
        unified_results = []
        
        # Process text results
        for result in text_results:
            unified_results.append(
                self._create_unified_result(result, 'text', weights['text'])
            )
        
        # Process table results
        for table_match in table_results:
            unified_results.append(
                self._create_unified_result_from_table(table_match, weights['table'])
            )
        
        # Process entity results
        for entity_match in entity_results:
            unified_results.append(
                self._create_unified_result_from_entity(entity_match, weights['entity'])
            )
        
        # Remove duplicates and re-rank
        deduplicated = self._remove_duplicate_chunks(unified_results)
        final_ranked = self._final_ranking(deduplicated)
        
        return final_ranked
```

### 4. **Advanced Scoring Mechanisms**

#### Semantic Similarity Enhancement
```python
class EnhancedSimilarityScoring:
    """Enhanced similarity scoring with structure awareness."""
    
    def calculate_enhanced_similarity(
        self,
        query_embedding: np.ndarray,
        chunk: TextChunk,
        query_analysis: QueryAnalysis
    ) -> float:
        """Calculate enhanced similarity score."""
        
        # Base cosine similarity
        base_similarity = self.cosine_similarity(query_embedding, chunk.embedding)
        
        # Structure relevance bonus
        structure_bonus = self._calculate_structure_bonus(chunk, query_analysis)
        
        # Entity alignment bonus
        entity_bonus = self._calculate_entity_alignment(chunk, query_analysis.entities)
        
        # Intent alignment bonus
        intent_bonus = self._calculate_intent_alignment(chunk, query_analysis.intent)
        
        # Combine scores with weights
        enhanced_score = (
            base_similarity * 0.6 +
            structure_bonus * 0.2 +
            entity_bonus * 0.15 +
            intent_bonus * 0.05
        )
        
        return min(enhanced_score, 1.0)  # Cap at 1.0
    
    def _calculate_structure_bonus(
        self, 
        chunk: TextChunk, 
        query_analysis: QueryAnalysis
    ) -> float:
        """Calculate bonus based on document structure relevance."""
        
        bonus = 0.0
        metadata = chunk.metadata
        
        # Document type alignment
        if 'document_type' in metadata:
            if self._is_document_type_relevant(metadata['document_type'], query_analysis):
                bonus += 0.1
        
        # Section type relevance
        if 'section_type' in metadata:
            section_relevance = self._calculate_section_relevance(
                metadata['section_type'], query_analysis.intent
            )
            bonus += section_relevance * 0.1
        
        # Table presence bonus for factual queries
        if query_analysis.intent == QueryIntent.FACTUAL:
            if metadata.get('chunk_type') == 'table':
                bonus += 0.15
        
        return bonus
    
    def _calculate_entity_alignment(
        self,
        chunk: TextChunk,
        query_entities: List[QueryEntity]
    ) -> float:
        """Calculate bonus based on entity alignment."""
        
        if not query_entities:
            return 0.0
        
        chunk_entities = chunk.metadata.get('overlapping_entities', [])
        if not chunk_entities:
            return 0.0
        
        # Calculate entity overlap
        query_entity_texts = {e.text.lower() for e in query_entities}
        chunk_entity_texts = {e.get('text', '').lower() for e in chunk_entities}
        
        overlap = len(query_entity_texts.intersection(chunk_entity_texts))
        max_possible = len(query_entity_texts)
        
        return overlap / max_possible if max_possible > 0 else 0.0
```

#### Context-Aware Ranking
```python
class ContextAwareRanking:
    """Advanced ranking using document context and relationships."""
    
    def rank_with_context(
        self,
        candidates: List[RankedChunk],
        query_analysis: QueryAnalysis,
        document_graph: Optional[DocumentGraph] = None
    ) -> List[RankedChunk]:
        """Re-rank candidates using contextual information."""
        
        for chunk in candidates:
            # Calculate context coherence
            context_score = self._calculate_context_coherence(
                chunk, candidates, document_graph
            )
            
            # Calculate authority score
            authority_score = self._calculate_authority_score(
                chunk, document_graph
            )
            
            # Calculate recency bonus
            recency_bonus = self._calculate_recency_bonus(chunk)
            
            # Update final score
            chunk.context_coherence_score = context_score
            chunk.final_rank_score = self._combine_ranking_scores(
                chunk, context_score, authority_score, recency_bonus
            )
        
        # Sort by final rank score
        return sorted(candidates, key=lambda x: x.final_rank_score, reverse=True)
    
    def _calculate_context_coherence(
        self,
        target_chunk: RankedChunk,
        all_candidates: List[RankedChunk],
        document_graph: Optional[DocumentGraph]
    ) -> float:
        """Calculate how well the chunk fits with other selected content."""
        
        coherence_score = 0.0
        
        # Check for document continuity
        same_doc_candidates = [
            c for c in all_candidates[:5]  # Top 5 candidates
            if c.chunk.metadata.get('document_id') == target_chunk.chunk.metadata.get('document_id')
        ]
        
        if len(same_doc_candidates) > 1:
            coherence_score += 0.1  # Bonus for document consistency
        
        # Check for section hierarchy coherence
        target_section = target_chunk.chunk.metadata.get('section_id')
        if target_section:
            related_sections = [
                c for c in same_doc_candidates
                if self._are_sections_related(
                    target_section, 
                    c.chunk.metadata.get('section_id'),
                    document_graph
                )
            ]
            coherence_score += len(related_sections) * 0.05
        
        return min(coherence_score, 0.3)  # Cap coherence bonus
```

### 5. **Specialized Search Modes**

#### Table-Aware Search
```python
class TableAwareSearch:
    """Specialized search for tabular content."""
    
    async def search_tables(
        self,
        query: str,
        query_entities: List[QueryEntity],
        similarity_threshold: float = 0.7
    ) -> List[TableMatch]:
        """Search specifically within table content."""
        
        # Find chunks that contain tables
        table_chunks = await self.vector_store.search_by_metadata({
            'chunk_type': 'table',
            'is_complete_table': True
        })
        
        table_matches = []
        
        for chunk in table_chunks:
            # Calculate table-specific similarity
            table_score = await self._calculate_table_similarity(
                query, chunk, query_entities
            )
            
            if table_score >= similarity_threshold:
                table_match = TableMatch(
                    chunk=chunk,
                    table_similarity_score=table_score,
                    matching_columns=self._find_matching_columns(chunk, query_entities),
                    matching_cells=self._find_matching_cells(chunk, query),
                    table_type=chunk.metadata.get('table_type', 'unknown')
                )
                table_matches.append(table_match)
        
        return sorted(table_matches, key=lambda x: x.table_similarity_score, reverse=True)
    
    async def _calculate_table_similarity(
        self,
        query: str,
        table_chunk: TextChunk,
        query_entities: List[QueryEntity]
    ) -> float:
        """Calculate similarity score specifically for table content."""
        
        # Base semantic similarity
        base_score = await self.embedding_service.calculate_similarity(
            query, table_chunk.content
        )
        
        # Header/column relevance
        headers = table_chunk.metadata.get('table_headers', [])
        header_relevance = self._calculate_header_relevance(query, headers)
        
        # Entity matching in table cells
        entity_match_score = self._calculate_table_entity_match(
            table_chunk, query_entities
        )
        
        # Table type bonus (e.g., financial tables for budget queries)
        type_bonus = self._calculate_table_type_bonus(
            query, table_chunk.metadata.get('table_type')
        )
        
        # Combine scores
        final_score = (
            base_score * 0.5 +
            header_relevance * 0.25 +
            entity_match_score * 0.15 +
            type_bonus * 0.1
        )
        
        return final_score
```

#### Entity-Focused Search
```python
class EntityFocusedSearch:
    """Search focused on entity relationships and context."""
    
    async def search_by_entities(
        self,
        query_entities: List[QueryEntity],
        entity_types: Optional[List[str]] = None,
        relationship_depth: int = 1
    ) -> List[EntityMatch]:
        """Search based on entity presence and relationships."""
        
        entity_matches = []
        
        for query_entity in query_entities:
            # Find chunks containing this entity
            matching_chunks = await self.vector_store.search_by_metadata({
                'overlapping_entities': {
                    'contains': query_entity.text
                }
            })
            
            for chunk in matching_chunks:
                entity_match = EntityMatch(
                    chunk=chunk,
                    matched_entity=query_entity,
                    entity_context=self._extract_entity_context(chunk, query_entity),
                    related_entities=self._find_related_entities(chunk, query_entity),
                    entity_prominence=self._calculate_entity_prominence(chunk, query_entity),
                    confidence_score=0.8
                )
                entity_matches.append(entity_match)
        
        # Remove duplicates and rank by entity prominence
        deduplicated = self._deduplicate_entity_matches(entity_matches)
        return sorted(deduplicated, key=lambda x: x.entity_prominence, reverse=True)
```

### 6. **Performance Optimization**

#### Efficient Search Pipeline
```python
class OptimizedRetrievalPipeline:
    """High-performance retrieval pipeline with caching and optimization."""
    
    def __init__(self, vector_store, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # 1-hour cache
        self.search_stats = SearchStatistics()
    
    async def retrieve_optimized(
        self,
        query_analysis: QueryAnalysis,
        top_k: int = 10,
        use_cache: bool = True
    ) -> RetrievalResult:
        """Optimized retrieval with caching and performance monitoring."""
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(query_analysis, top_k)
        if use_cache and cache_key in self.cache:
            self.search_stats.cache_hits += 1
            return self.cache[cache_key]
        
        # Determine optimal search strategy
        strategy = self._select_optimal_strategy(query_analysis)
        
        # Parallel search across different modes if applicable
        if strategy.search_modes and len(strategy.search_modes) > 1:
            results = await self._parallel_multi_modal_search(query_analysis, strategy)
        else:
            results = await self._single_mode_search(query_analysis, strategy)
        
        # Apply post-processing and ranking
        final_results = await self._post_process_results(results, query_analysis)
        
        # Update performance statistics
        processing_time = time.time() - start_time
        self.search_stats.update(processing_time, len(final_results.ranked_chunks))
        
        # Cache results
        if use_cache:
            self.cache[cache_key] = final_results
        
        return final_results
    
    async def _parallel_multi_modal_search(
        self,
        query_analysis: QueryAnalysis,
        strategy: RetrievalStrategy
    ) -> List[RankedChunk]:
        """Execute parallel searches across multiple modes."""
        
        search_tasks = []
        
        if SearchMode.TEXT in strategy.search_modes:
            search_tasks.append(
                self._search_text_mode(query_analysis, strategy)
            )
        
        if SearchMode.TABLE in strategy.search_modes:
            search_tasks.append(
                self._search_table_mode(query_analysis, strategy)
            )
        
        if SearchMode.ENTITY in strategy.search_modes:
            search_tasks.append(
                self._search_entity_mode(query_analysis, strategy)
            )
        
        # Execute all searches in parallel
        search_results = await asyncio.gather(*search_tasks)
        
        # Combine and deduplicate results
        combined_results = []
        for results in search_results:
            combined_results.extend(results)
        
        return self._deduplicate_and_merge(combined_results)
```

## Implementation Timeline

### Week 1: Core Retrieval Infrastructure
- Set up IntelligentRetriever service structure
- Implement basic multi-modal search capability
- Create data models and enums
- Build foundation for structure-aware filtering

### Week 2: Advanced Scoring and Ranking
- Implement enhanced similarity scoring
- Build context-aware ranking system
- Create intent-based retrieval strategies
- Add document-type-aware retrieval logic

### Week 3: Specialized Search Modes
- Implement table-aware search
- Build entity-focused search capabilities
- Create multi-modal result fusion
- Add performance optimization features

### Week 4: Integration and Testing
- Integrate with QueryAnalyzer and DocumentAnalyzer
- Comprehensive testing with real documents
- Performance benchmarking and optimization
- Documentation and example implementations

## Integration Examples

### Complete RAG Pipeline Integration
```python
async def enhanced_rag_pipeline(
    user_query: str,
    conversation_context: Optional[List[str]] = None
) -> RAGResponse:
    """Complete enhanced RAG pipeline."""
    
    # Phase 1: Query Analysis
    query_analyzer = QueryAnalyzer()
    query_analysis = await query_analyzer.analyze_query(
        user_query, 
        context={'conversation': conversation_context}
    )
    
    # Phase 2: Intelligent Retrieval
    intelligent_retriever = IntelligentRetriever()
    retrieval_result = await intelligent_retriever.retrieve_intelligent(
        query_analysis,
        top_k=10
    )
    
    # Phase 3: Context Preparation
    context_chunks = [chunk.chunk for chunk in retrieval_result.ranked_chunks]
    context_text = prepare_context_text(context_chunks, query_analysis)
    
    # Phase 4: Response Generation (existing LLM integration)
    llm_response = await generate_llm_response(
        user_query,
        context_text,
        query_analysis.intent
    )
    
    return RAGResponse(
        query=user_query,
        query_analysis=query_analysis,
        retrieval_result=retrieval_result,
        response=llm_response,
        sources=extract_sources(context_chunks),
        confidence=calculate_overall_confidence(retrieval_result, llm_response)
    )
```

## Success Metrics

### Performance Targets
- **Retrieval Speed**: < 500ms for standard queries, < 2s for complex multi-modal queries
- **Accuracy Improvement**: 30%+ improvement over basic similarity search
- **Context Relevance**: 85%+ of retrieved chunks directly relevant to query
- **Multi-Modal Effectiveness**: 90%+ success rate in finding relevant tables/entities when present

### Quality Improvements
- **Answer Quality**: 25-40% improvement in RAG response quality
- **Source Accuracy**: 95%+ of cited sources contain relevant information
- **Context Completeness**: Reduced need for follow-up queries by 60%+

## Testing Strategy

### Comprehensive Test Suite
- **Unit Tests**: Individual component functionality
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Speed and memory usage benchmarks
- **Quality Tests**: Retrieval accuracy and relevance assessment

### Test Data Requirements
- **Diverse Document Types**: Government, technical, business documents
- **Complex Queries**: Multi-intent, entity-rich, analytical queries
- **Edge Cases**: Ambiguous queries, rare entities, cross-document relationships

This completes the comprehensive plan for Phase 3.3d, which will deliver a state-of-the-art intelligent retrieval system that fully leverages our enhanced document processing and query analysis capabilities! 🎯
