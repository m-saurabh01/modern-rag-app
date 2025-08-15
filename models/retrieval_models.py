"""
Data models for intelligent retrieval system.

This module contains all data structures used by the IntelligentRetriever service
for Phase 3.3d implementation with advanced multi-modal search capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import time
from datetime import datetime

from .query_models import QueryAnalysis, QueryEntity


# ==================== ENUMS ====================

class SearchMode(Enum):
    """Different search modes for multi-modal retrieval."""
    TEXT = "text"                    # Standard semantic text search
    TABLE = "table"                  # Table-specific search with structure analysis
    ENTITY = "entity"                # Entity-focused search with relationships
    STRUCTURE = "structure"          # Document structure-aware search
    CONTEXT = "context"              # Context-aware search with relationships
    HYBRID = "hybrid"                # Combined multi-modal search


class RetrievalMode(Enum):
    """Performance modes for retrieval processing."""
    SPEED_OPTIMIZED = "speed"        # <300ms, simpler ranking, aggressive caching
    BALANCED = "balanced"            # <500ms, moderate complexity (DEFAULT)
    ACCURACY_OPTIMIZED = "accuracy"  # <1000ms, full analysis and comprehensive ranking


class RerankingComplexity(Enum):
    """Re-ranking complexity levels."""
    BASIC = "basic"                  # 3 factors: similarity + intent + entity
    ADVANCED = "advanced"            # 6 factors: + structure + coherence + authority
    COMPREHENSIVE = "comprehensive"  # 9+ factors: + relationships + temporal + references


class ContextExpansion(Enum):
    """Context expansion strategies."""
    MINIMAL = "minimal"              # Individual chunks only
    DOCUMENT_CONTEXT = "document"    # Same document/section coherence
    CROSS_DOCUMENT = "cross_doc"     # Cross-document relationships


class FusionStrategy(Enum):
    """Multi-modal result fusion strategies."""
    WEIGHTED_AVERAGE = "weighted_avg"     # Simple weighted combination
    RANK_FUSION = "rank_fusion"          # Reciprocal rank fusion
    LEARNED_FUSION = "learned"           # ML-based fusion (future)


# ==================== CORE DATA STRUCTURES ====================

@dataclass
class RetrievalConfig:
    """Configuration for intelligent retrieval behavior."""
    
    # Performance settings
    retrieval_mode: RetrievalMode = RetrievalMode.BALANCED
    target_response_time_ms: float = 500.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 1800  # 30 minutes
    
    # Multi-modal weighting (query-adaptive by default)
    enable_query_adaptive_weighting: bool = True
    default_modal_weights: Dict[str, float] = field(default_factory=lambda: {
        'text': 0.6, 'table': 0.25, 'entity': 0.15
    })
    
    # Structure influence
    structure_influence_weight: float = 0.25  # 25% influence (moderate)
    enable_structure_filtering: bool = True
    
    # Re-ranking complexity
    reranking_complexity: RerankingComplexity = RerankingComplexity.ADVANCED
    enable_complexity_switching: bool = True
    
    # Table analysis
    enable_semantic_table_analysis: bool = True
    table_header_weight: float = 0.3
    table_cell_analysis: bool = True
    
    # Context expansion
    context_expansion: ContextExpansion = ContextExpansion.DOCUMENT_CONTEXT
    enable_cross_document_context: bool = True
    max_context_chunks: int = 5
    
    # Performance optimization
    parallel_search_enabled: bool = True
    result_deduplication: bool = True
    similarity_threshold: float = 0.6


@dataclass
class QueryAdaptiveWeights:
    """Dynamic weights based on query analysis."""
    text_weight: float
    table_weight: float
    entity_weight: float
    structure_weight: float
    temporal_weight: float = 0.0
    
    def normalize(self) -> 'QueryAdaptiveWeights':
        """Ensure weights sum to 1.0."""
        total = (self.text_weight + self.table_weight + 
                self.entity_weight + self.structure_weight + self.temporal_weight)
        
        if total == 0:
            return QueryAdaptiveWeights(0.6, 0.25, 0.15, 0.0, 0.0)
            
        return QueryAdaptiveWeights(
            text_weight=self.text_weight / total,
            table_weight=self.table_weight / total,
            entity_weight=self.entity_weight / total,
            structure_weight=self.structure_weight / total,
            temporal_weight=self.temporal_weight / total
        )


@dataclass
class TableMatch:
    """Detailed table search result with semantic analysis."""
    chunk: 'TextChunk'  # Forward reference
    table_similarity_score: float
    matching_columns: List[str]
    matching_cells: List[Dict[str, Any]]
    table_type: str
    header_relevance_score: float
    entity_alignment_score: float
    data_type_confidence: float
    table_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityMatch:
    """Entity-focused search result with relationship context."""
    chunk: 'TextChunk'  # Forward reference
    matched_entity: QueryEntity
    entity_context: str
    related_entities: List[Dict[str, Any]]
    entity_prominence: float
    relationship_strength: float
    confidence_score: float
    entity_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructureMatch:
    """Structure-aware search result."""
    chunk: 'TextChunk'  # Forward reference
    structure_relevance_score: float
    section_type_alignment: float
    hierarchy_bonus: float
    document_type_match: float
    structure_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedChunk:
    """Chunk with comprehensive ranking information."""
    chunk: 'TextChunk'  # Forward reference
    
    # Core similarity scores
    base_similarity_score: float
    structure_relevance_score: float
    entity_alignment_score: float
    intent_alignment_score: float
    context_coherence_score: float
    
    # Advanced scoring factors (for COMPREHENSIVE mode)
    authority_score: float = 0.0
    recency_score: float = 0.0
    cross_reference_score: float = 0.0
    temporal_relevance_score: float = 0.0
    
    # Final ranking
    final_rank_score: float = 0.0
    rank_position: int = 0
    ranking_explanation: str = ""
    
    # Performance metadata
    processing_time_ms: float = 0.0
    ranking_factors_used: List[str] = field(default_factory=list)


@dataclass
class MultiModalResult:
    """Complete multi-modal search result with fusion details."""
    
    # Individual modal results
    text_results: List[RankedChunk]
    table_results: List[TableMatch]
    entity_results: List[EntityMatch]
    structure_results: List[StructureMatch]
    
    # Fused results
    combined_results: List[RankedChunk]
    
    # Fusion metadata
    modal_weights: QueryAdaptiveWeights
    fusion_strategy: FusionStrategy
    fusion_confidence: float
    
    # Performance metrics
    total_candidates: int
    processing_time_ms: float
    cache_hit_rate: float = 0.0


@dataclass
class RetrievalResult:
    """Complete retrieval result with comprehensive metadata."""
    
    # Core result data
    query_id: str
    query_analysis: QueryAnalysis
    ranked_chunks: List[RankedChunk]
    retrieval_strategy: str
    config_used: RetrievalConfig
    total_candidates: int
    processing_time: float
    confidence_score: float
    
    # Optional fields with defaults
    multi_modal_result: Optional[MultiModalResult] = None
    
    # Context and relationships
    document_contexts: Dict[str, List[str]] = field(default_factory=dict)
    cross_document_relationships: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality metrics
    precision_estimate: float = 0.0
    coverage_score: float = 0.0
    
    # Metadata and debugging
    metadata: Dict[str, Any] = field(default_factory=dict)
    debug_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default query_id if not provided."""
        if not self.query_id:
            self.query_id = f"retr_{int(time.time() * 1000)}"


@dataclass
class ContextRelationship:
    """Relationship between chunks for context expansion."""
    source_chunk_id: str
    target_chunk_id: str
    relationship_type: str  # 'same_document', 'cross_reference', 'temporal', 'causal'
    strength: float  # 0.0 to 1.0
    evidence: List[str]  # Supporting evidence for the relationship
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentGraph:
    """Graph representation of document relationships."""
    documents: Dict[str, Dict[str, Any]]
    relationships: List[ContextRelationship]
    entity_connections: Dict[str, List[str]]
    temporal_ordering: Dict[str, datetime]
    authority_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class SearchStatistics:
    """Performance and quality statistics for retrieval."""
    
    # Performance metrics
    total_searches: int = 0
    average_response_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Quality metrics
    precision_scores: List[float] = field(default_factory=list)
    recall_estimates: List[float] = field(default_factory=list)
    user_satisfaction: List[float] = field(default_factory=list)
    
    # Mode usage
    mode_usage: Dict[str, int] = field(default_factory=dict)
    complexity_switches: Dict[str, int] = field(default_factory=dict)
    
    # Error tracking
    error_count: int = 0
    fallback_usage: Dict[str, int] = field(default_factory=dict)
    
    def update(self, response_time: float, result_count: int, 
              precision: Optional[float] = None):
        """Update statistics with new search result."""
        self.total_searches += 1
        
        # Update running average
        if self.total_searches == 1:
            self.average_response_time_ms = response_time
        else:
            alpha = 0.1  # Smoothing factor
            self.average_response_time_ms = (
                alpha * response_time + 
                (1 - alpha) * self.average_response_time_ms
            )
        
        if precision is not None:
            self.precision_scores.append(precision)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0
    
    @property
    def average_precision(self) -> float:
        """Calculate average precision."""
        return sum(self.precision_scores) / len(self.precision_scores) if self.precision_scores else 0.0


# ==================== CONFIGURATION HELPERS ====================

def create_speed_optimized_config() -> RetrievalConfig:
    """Create configuration optimized for speed (<300ms)."""
    return RetrievalConfig(
        retrieval_mode=RetrievalMode.SPEED_OPTIMIZED,
        target_response_time_ms=300.0,
        reranking_complexity=RerankingComplexity.BASIC,
        structure_influence_weight=0.15,
        enable_semantic_table_analysis=False,
        context_expansion=ContextExpansion.MINIMAL,
        parallel_search_enabled=True,
        similarity_threshold=0.7  # Higher threshold for speed
    )


def create_balanced_config() -> RetrievalConfig:
    """Create balanced configuration (<500ms) - DEFAULT."""
    return RetrievalConfig(
        retrieval_mode=RetrievalMode.BALANCED,
        target_response_time_ms=500.0,
        reranking_complexity=RerankingComplexity.ADVANCED,
        structure_influence_weight=0.25,
        enable_semantic_table_analysis=True,
        context_expansion=ContextExpansion.DOCUMENT_CONTEXT,
        enable_cross_document_context=True,
        parallel_search_enabled=True
    )


def create_accuracy_optimized_config() -> RetrievalConfig:
    """Create configuration optimized for accuracy (<1000ms)."""
    return RetrievalConfig(
        retrieval_mode=RetrievalMode.ACCURACY_OPTIMIZED,
        target_response_time_ms=1000.0,
        reranking_complexity=RerankingComplexity.COMPREHENSIVE,
        structure_influence_weight=0.3,
        enable_semantic_table_analysis=True,
        table_cell_analysis=True,
        context_expansion=ContextExpansion.CROSS_DOCUMENT,
        enable_cross_document_context=True,
        max_context_chunks=10,
        parallel_search_enabled=True,
        similarity_threshold=0.5  # Lower threshold for higher recall
    )


# ==================== FORWARD REFERENCE RESOLUTION ====================

# This will be resolved when TextChunk is imported
# For now, we'll use a placeholder type
try:
    from ..services.chunking_service import TextChunk
except ImportError:
    # Placeholder for type hints during development
    class TextChunk:
        pass
