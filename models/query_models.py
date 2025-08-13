"""
Query Analysis Data Models

This module defines the data structures used for query analysis in the Modern RAG system.
These models support intelligent query understanding, entity extraction, and retrieval optimization.

Key Features:
- Query intent classification with confidence scoring
- Entity extraction with domain-specific types
- Context-aware query expansion with precision focus
- Configurable processing modes and caching options
- Extensible design for custom intent/entity types
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4


class QueryIntent(Enum):
    """
    Query intent classification for optimized retrieval strategies.
    
    These categories determine how the retrieval system should approach
    finding and ranking relevant information.
    
    **Extensibility Note**: To add new intent types:
    1. Add new enum value here
    2. Update QueryAnalyzer._classify_intent_patterns() method
    3. Update IntelligentRetriever strategy selection logic
    4. Add test cases in test_query_analyzer.py
    """
    FACTUAL = "factual"              # Seeking specific facts or data points
    ANALYTICAL = "analytical"        # Requesting analysis or interpretation  
    COMPARATIVE = "comparative"      # Comparing entities, concepts, or options
    PROCEDURAL = "procedural"        # Looking for how-to information or processes
    EXPLORATORY = "exploratory"      # Open-ended research or discovery
    VERIFICATION = "verification"    # Confirming or validating information


class QuestionType(Enum):
    """
    Question structure classification for answer optimization.
    
    **Extensibility Note**: To add new question types:
    1. Add new enum value here
    2. Update QueryAnalyzer._detect_question_type_patterns() method
    3. Update answer generation strategies accordingly
    """
    WHO = "who"                      # Person-focused queries
    WHAT = "what"                    # Definition or explanation queries
    WHERE = "where"                  # Location-based queries
    WHEN = "when"                    # Time-based queries
    WHY = "why"                      # Causation or reasoning queries
    HOW = "how"                      # Process or method queries
    HOW_MUCH = "how_much"           # Quantitative queries (amount)
    HOW_MANY = "how_many"           # Quantitative queries (count)
    YES_NO = "yes_no"               # Boolean queries
    OPEN_ENDED = "open_ended"       # Complex analytical queries


class EntityType(Enum):
    """
    Entity types extracted from queries for focused retrieval.
    
    **Extensibility Note**: To add domain-specific entity types:
    1. Add new enum value here
    2. Update QueryAnalyzer._extract_entities_patterns() method
    3. Add corresponding patterns in entity_patterns dictionary
    4. Consider spaCy model updates for better recognition
    """
    PERSON = "person"                # Names, titles, roles
    ORGANIZATION = "organization"    # Departments, companies, agencies
    LOCATION = "location"           # Places, addresses, regions
    DATE_TIME = "date_time"         # Dates, periods, timeframes
    DOCUMENT_TYPE = "document_type" # Reports, policies, forms
    DOMAIN_TERM = "domain_term"     # Technical/business terminology
    NUMERIC = "numeric"             # Numbers, percentages, amounts
    CURRENCY = "currency"           # Financial amounts
    EMAIL = "email"                 # Email addresses
    PHONE = "phone"                 # Phone numbers
    URL = "url"                     # Web addresses
    POLICY_ID = "policy_id"         # Policy/regulation identifiers
    PROJECT_ID = "project_id"       # Project identifiers


class ProcessingMode(Enum):
    """
    Query processing performance vs accuracy trade-offs.
    
    Default: BALANCED - Optimal mix of speed and accuracy
    """
    FAST = "fast"                   # Pattern-based, < 50ms
    BALANCED = "balanced"           # Mixed NLP + patterns, < 200ms (DEFAULT)
    COMPREHENSIVE = "comprehensive" # Full NLP analysis, < 500ms


class RetrievalStrategy(Enum):
    """
    Suggested retrieval strategies based on query analysis.
    
    These hints guide the IntelligentRetriever in Phase 3.3d
    """
    ENTITY_FOCUSED = "entity_focused"        # Focus on entity matching
    SEMANTIC_SEARCH = "semantic_search"      # Broad semantic similarity
    STRUCTURE_AWARE = "structure_aware"      # Leverage document structure
    TABLE_PRIORITY = "table_priority"        # Prioritize tabular data
    TEMPORAL_FILTERING = "temporal_filtering" # Filter by time periods
    MULTI_DOCUMENT = "multi_document"        # Cross-document comparison


@dataclass
class QueryEntity:
    """
    Entity extracted from user query with confidence and context.
    
    **Usage Example**:
    ```python
    entity = QueryEntity(
        text="Department of Health",
        entity_type=EntityType.ORGANIZATION,
        start_position=15,
        end_position=34,
        confidence=0.95,
        context="Find documents from Department of Health",
        variants=["DOH", "Health Department"],
        domain_relevance=0.9
    )
    ```
    """
    text: str                          # Original entity text
    entity_type: EntityType            # Classified entity type
    start_position: int                # Character start position in query
    end_position: int                  # Character end position in query
    confidence: float                  # Extraction confidence (0.0-1.0)
    context: str                       # Surrounding query context
    variants: List[str] = field(default_factory=list)  # Alternative representations
    domain_relevance: float = 1.0      # Relevance to domain (0.0-1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional properties


@dataclass
class QueryExpansion:
    """
    Query expansion with related terms for improved retrieval.
    
    **Conservative Expansion Strategy**: Focus on precision over recall
    - Limited synonym selection (top 3 per term)
    - High-confidence domain terms only
    - Contextually relevant temporal expansions
    
    **Usage Example**:
    ```python
    expansion = QueryExpansion(
        original_terms=["budget", "allocation"],
        synonyms={"budget": ["funding", "allocation"], "allocation": ["distribution"]},
        related_concepts=["financial planning", "resource management"],
        domain_terms=["fiscal year", "appropriations"],
        expansion_confidence=0.8
    )
    ```
    """
    original_terms: List[str]                      # Key terms from original query
    synonyms: Dict[str, List[str]] = field(default_factory=dict)  # Term -> synonyms
    related_concepts: List[str] = field(default_factory=list)     # Contextually related terms
    domain_terms: List[str] = field(default_factory=list)        # Domain-specific terminology
    temporal_context: List[str] = field(default_factory=list)    # Time-related expansions
    expansion_confidence: float = 0.0              # Overall expansion quality (0.0-1.0)
    suggested_filters: Dict[str, Any] = field(default_factory=dict)  # Retrieval filters
    
    # Conservative expansion limits
    max_synonyms_per_term: int = 3
    max_related_concepts: int = 5
    max_domain_terms: int = 7
    min_confidence_threshold: float = 0.6


@dataclass
class QueryAnalysis:
    """
    Complete query analysis result with all enhancement features.
    
    This is the primary output of QueryAnalyzer.analyze_query() and contains
    all information needed for intelligent retrieval optimization.
    
    **Usage Example**:
    ```python
    analysis = await query_analyzer.analyze_query(
        "What was the budget allocation for IT department in Q2 2024?"
    )
    
    # Access analysis results
    print(f"Intent: {analysis.intent}")
    print(f"Entities: {[e.text for e in analysis.entities]}")
    print(f"Suggested strategy: {analysis.suggested_strategies[0]}")
    ```
    """
    query_id: str                      # Unique query identifier
    original_query: str                # Original user query text
    processed_query: str               # Cleaned/normalized query
    intent: QueryIntent                # Classified query intent
    question_type: QuestionType        # Question structure type
    entities: List[QueryEntity]        # Extracted entities with metadata
    expansion: QueryExpansion          # Query expansion suggestions
    complexity_score: float           # Query complexity (0.0-1.0, higher = more complex)
    confidence_score: float           # Overall analysis confidence (0.0-1.0)
    suggested_strategies: List[RetrievalStrategy]  # Recommended retrieval approaches
    processing_time: float            # Analysis time in milliseconds
    processing_mode: ProcessingMode   # Processing mode used
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional analysis data
    
    def __post_init__(self):
        """Generate unique query ID if not provided."""
        if not self.query_id:
            self.query_id = str(uuid4())


@dataclass
class QueryCache:
    """
    Optional caching configuration for query analysis results.
    
    **Caching Strategy**: Conservative approach for production environments
    - Cache analysis results for identical queries
    - Cache expansion lookups to reduce WordNet/spaCy calls
    - TTL-based expiration to handle evolving document knowledge
    
    **Usage Example**:
    ```python
    cache_config = QueryCache(
        enabled=True,
        ttl_seconds=3600,  # 1 hour cache
        max_entries=1000
    )
    
    query_analyzer = QueryAnalyzer(cache_config=cache_config)
    ```
    
    **Performance Impact**:
    - Cache hits: ~5ms response time
    - Cache misses: Normal processing time
    - Memory usage: ~50KB per 100 cached queries
    """
    enabled: bool = False              # Enable/disable caching
    ttl_seconds: int = 3600           # Time-to-live in seconds (default: 1 hour)
    max_entries: int = 1000           # Maximum cache entries
    cache_expansion_lookups: bool = True  # Cache synonym/concept lookups
    cache_entity_extractions: bool = True  # Cache entity extraction results
    memory_limit_mb: int = 50         # Maximum cache memory usage (MB)


@dataclass
class QueryAnalyzerConfig:
    """
    Configuration for QueryAnalyzer service behavior.
    
    **Default Configuration**: Balanced performance with conservative expansion
    
    **Customization Example**:
    ```python
    config = QueryAnalyzerConfig(
        processing_mode=ProcessingMode.COMPREHENSIVE,
        enable_query_expansion=True,
        expansion_aggressiveness=0.3,  # Conservative
        entity_confidence_threshold=0.7,
        cache_config=QueryCache(enabled=True)
    )
    
    query_analyzer = QueryAnalyzer(config=config)
    ```
    """
    processing_mode: ProcessingMode = ProcessingMode.BALANCED  # Default processing mode
    enable_query_expansion: bool = True         # Enable query expansion
    expansion_aggressiveness: float = 0.3       # Conservative expansion (0.0-1.0)
    entity_confidence_threshold: float = 0.6    # Minimum entity confidence
    intent_confidence_threshold: float = 0.7    # Minimum intent confidence
    max_processing_time_ms: float = 500.0      # Maximum processing time limit
    fallback_to_patterns: bool = True          # Use pattern fallbacks if NLP fails
    integrate_document_knowledge: bool = True   # Use DocumentAnalyzer knowledge
    integrate_embedding_similarity: bool = False  # Compare with document embeddings
    cache_config: Optional[QueryCache] = None   # Optional caching configuration
    
    # Extensibility settings
    custom_intent_patterns: Dict[str, List[str]] = field(default_factory=dict)
    custom_entity_patterns: Dict[str, List[str]] = field(default_factory=dict)
    domain_specific_expansions: Dict[str, List[str]] = field(default_factory=dict)


# Type aliases for convenience
QueryAnalysisResult = QueryAnalysis
EntityList = List[QueryEntity]
IntentClassification = QueryIntent
QuestionClassification = QuestionType
