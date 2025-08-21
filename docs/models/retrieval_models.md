# Retrieval Models Documentation

## 📋 Overview

The retrieval models module defines Pydantic data structures for search results, ranking metadata, and retrieval configuration. These models ensure type safety and data validation throughout the intelligent retrieval pipeline.

## 🏗️ Core Models

### **RetrievalResult**
Main result object containing search results and metadata:

```python
@dataclass
class RetrievalResult:
    """Complete retrieval result with ranking and metadata."""
    
    # Query information
    query_id: str
    original_query: str
    processed_query: str
    
    # Retrieved documents
    documents: List[DocumentResult]
    total_found: int
    
    # Retrieval metadata
    retrieval_metadata: RetrievalMetadata
    ranking_info: RankingInfo
    
    # Performance metrics
    retrieval_time: float
    processing_mode: PerformanceMode
    
    # Quality metrics
    relevance_scores: List[float]
    confidence_score: float
    
    # Pagination
    page: int = 1
    page_size: int = 10
    has_more: bool = False
    
    timestamp: datetime = field(default_factory=datetime.now)
```

### **DocumentResult**
Individual document result with relevance scoring:

```python
@dataclass
class DocumentResult:
    """Individual document in retrieval results."""
    
    # Document identification
    document_id: str
    chunk_id: str
    source: str
    
    # Content
    content: str
    title: Optional[str] = None
    excerpt: Optional[str] = None  # Highlighted excerpt
    
    # Metadata
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    section: Optional[str] = None
    
    # Scoring
    relevance_score: float        # Final relevance score (0.0-1.0)
    semantic_score: float         # Semantic similarity score
    keyword_score: float          # Keyword matching score
    structure_score: float        # Document structure relevance
    
    # Ranking factors
    ranking_factors: Dict[str, float]
    
    # Content analysis
    content_type: ContentType
    quality_score: float
    
    # Highlighting
    highlights: List[TextHighlight] = field(default_factory=list)
    
    # Context
    context_before: Optional[str] = None
    context_after: Optional[str] = None
```

### **RetrievalMetadata**
Detailed retrieval processing information:

```python
@dataclass
class RetrievalMetadata:
    """Retrieval processing metadata and statistics."""
    
    # Strategy information
    strategy_used: RetrievalStrategy
    performance_mode: PerformanceMode
    reranking_complexity: RerankingComplexity
    
    # Search details
    embedding_model: str
    vector_search_params: Dict[str, Any]
    filters_applied: Dict[str, Any]
    
    # Performance metrics
    stages: Dict[str, float]       # Time per processing stage
    total_candidates: int          # Total documents considered
    filtered_candidates: int       # After pre-filtering
    reranked_results: int         # After re-ranking
    
    # Query analysis integration
    query_analysis_used: bool
    entities_used: List[str]
    expansion_terms_used: List[str]
    
    # Caching information
    cache_hit: bool = False
    cache_source: Optional[str] = None
    
    # Error handling
    fallback_used: bool = False
    warnings: List[str] = field(default_factory=list)
```

## 🎯 Performance and Strategy Enums

### **PerformanceMode**
Retrieval performance optimization levels:

```python
class PerformanceMode(str, Enum):
    """Retrieval performance modes with speed/accuracy trade-offs."""
    
    SPEED = "speed"           # < 300ms, basic accuracy
    BALANCED = "balanced"     # < 500ms, good accuracy (DEFAULT)
    ACCURACY = "accuracy"     # < 1000ms, maximum accuracy
```

### **RerankingComplexity**
Re-ranking sophistication levels:

```python
class RerankingComplexity(str, Enum):
    """Re-ranking complexity levels."""
    
    BASIC = "basic"               # 3 factors, ~50ms overhead
    ADVANCED = "advanced"         # 6 factors, ~150ms overhead
    COMPREHENSIVE = "comprehensive"  # 9+ factors, ~400ms overhead
```

### **ContentType**
Classification of retrieved content:

```python
class ContentType(str, Enum):
    """Content type classification for retrieved documents."""
    
    TEXT = "text"                 # Regular text content
    TABLE = "table"               # Tabular data
    HEADER = "header"             # Section headers
    LIST = "list"                 # List items
    CAPTION = "caption"           # Image/table captions
    FOOTNOTE = "footnote"         # Footnotes and references
    METADATA = "metadata"         # Document metadata
    MIXED = "mixed"               # Mixed content types
```

## 📊 Ranking and Scoring

### **RankingInfo**
Detailed ranking algorithm information:

```python
@dataclass
class RankingInfo:
    """Ranking algorithm information and factor weights."""
    
    # Algorithm details
    algorithm_used: str
    complexity_level: RerankingComplexity
    
    # Factor weights
    semantic_weight: float = 0.4
    keyword_weight: float = 0.2
    structure_weight: float = 0.2
    quality_weight: float = 0.1
    freshness_weight: float = 0.1
    
    # Advanced factors (comprehensive mode)
    entity_overlap_weight: float = 0.0
    cross_document_weight: float = 0.0
    table_relevance_weight: float = 0.0
    
    # Normalization
    score_normalization: str = "min_max"  # "min_max", "z_score", "none"
    
    # Quality thresholds
    min_relevance_threshold: float = 0.3
    quality_threshold: float = 0.5
```

### **TextHighlight**
Text highlighting for search results:

```python
@dataclass
class TextHighlight:
    """Text highlighting information for search results."""
    
    text: str                     # Highlighted text
    start_pos: int               # Start position in content
    end_pos: int                 # End position in content
    highlight_type: HighlightType # Type of highlight
    relevance: float             # Relevance of this highlight (0.0-1.0)
    context: str                 # Surrounding context
```

### **HighlightType**
Types of text highlighting:

```python
class HighlightType(str, Enum):
    """Text highlighting types."""
    
    EXACT_MATCH = "exact_match"           # Exact query term match
    SEMANTIC_MATCH = "semantic_match"     # Semantically similar content
    ENTITY_MATCH = "entity_match"         # Named entity match
    EXPANSION_MATCH = "expansion_match"   # Query expansion term match
    KEYWORD_MATCH = "keyword_match"       # Important keyword match
```

## ⚙️ Configuration Models

### **RetrievalConfig**
Configuration for retrieval behavior:

```python
@dataclass
class RetrievalConfig:
    """Configuration for intelligent retrieval."""
    
    # Performance settings
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    max_response_time: float = 0.5
    
    # Search parameters
    max_results: int = 20
    min_relevance_score: float = 0.3
    enable_reranking: bool = True
    reranking_complexity: RerankingComplexity = RerankingComplexity.ADVANCED
    
    # Multi-modal settings
    text_weight: float = 0.6
    table_weight: float = 0.25
    entity_weight: float = 0.15
    
    # Context expansion
    context_expansion: ContextExpansionStrategy = ContextExpansionStrategy.HYBRID
    max_context_chunks: int = 3
    
    # Filtering
    enable_metadata_filtering: bool = True
    enable_quality_filtering: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    
    # Highlighting
    enable_highlighting: bool = True
    max_highlights_per_result: int = 3
    highlight_context_chars: int = 100
```

### **ContextExpansionStrategy**
Context expansion approaches:

```python
class ContextExpansionStrategy(str, Enum):
    """Context expansion strategies."""
    
    NONE = "none"                     # No context expansion
    DOCUMENT_LEVEL = "document_level" # Same document context
    CROSS_DOCUMENT = "cross_document" # Cross-document context
    HYBRID = "hybrid"                 # Mixed approach (recommended)
```

## 🔍 Search and Filter Models

### **SearchFilter**
Advanced search filtering options:

```python
@dataclass
class SearchFilter:
    """Advanced search filtering configuration."""
    
    # Document metadata filters
    document_types: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    date_range: Optional[DateRange] = None
    
    # Content filters
    content_types: Optional[List[ContentType]] = None
    min_quality_score: Optional[float] = None
    languages: Optional[List[str]] = None
    
    # Structural filters
    sections: Optional[List[str]] = None
    page_ranges: Optional[List[PageRange]] = None
    
    # Entity filters
    required_entities: Optional[List[str]] = None
    excluded_entities: Optional[List[str]] = None
    
    # Custom metadata filters
    custom_filters: Dict[str, Any] = field(default_factory=dict)
```

### **DateRange and PageRange**
Range filtering utilities:

```python
@dataclass
class DateRange:
    """Date range filtering."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    relative_days: Optional[int] = None  # e.g., last 30 days

@dataclass
class PageRange:
    """Page range filtering."""
    start_page: int
    end_page: int
    document_id: Optional[str] = None  # Specific document
```

## 📈 Performance Models

### **RetrievalPerformance**
Performance monitoring and metrics:

```python
@dataclass
class RetrievalPerformance:
    """Retrieval performance metrics."""
    
    # Timing metrics
    total_time: float
    embedding_time: float
    search_time: float
    reranking_time: float
    filtering_time: float
    
    # Throughput metrics
    documents_per_second: float
    queries_per_second: float
    
    # Quality metrics
    precision_at_k: Dict[int, float]  # Precision at different k values
    recall_at_k: Dict[int, float]     # Recall at different k values
    mrr: float                        # Mean Reciprocal Rank
    ndcg: float                       # Normalized Discounted Cumulative Gain
    
    # Resource usage
    memory_usage_mb: float
    cpu_usage_percent: float
    
    # Cache statistics
    cache_hit_rate: float
    cache_miss_rate: float
```

## 🔄 Usage Examples

### **Basic Retrieval**
```python
from models.retrieval_models import RetrievalConfig, RetrievalResult

# Configure retrieval
config = RetrievalConfig(
    performance_mode=PerformanceMode.BALANCED,
    max_results=10,
    enable_highlighting=True
)

# Perform retrieval (type-safe)
result: RetrievalResult = await intelligent_retriever.retrieve(
    query="AI research methods",
    config=config
)

# Access typed results
documents: List[DocumentResult] = result.documents
metadata: RetrievalMetadata = result.retrieval_metadata
```

### **Advanced Filtering**
```python
# Create complex filters
search_filter = SearchFilter(
    document_types=["research_paper", "technical_report"],
    date_range=DateRange(relative_days=365),  # Last year
    min_quality_score=0.8,
    required_entities=["machine learning", "neural networks"]
)

# Apply filters to retrieval
result = await intelligent_retriever.retrieve(
    query="deep learning techniques",
    filters=search_filter,
    config=config
)
```

### **Result Processing**
```python
# Process results with type safety
for doc in result.documents:
    print(f"Document: {doc.source}")
    print(f"Relevance: {doc.relevance_score:.3f}")
    print(f"Content Type: {doc.content_type}")
    
    # Process highlights
    for highlight in doc.highlights:
        print(f"Highlight: {highlight.text}")
        print(f"Type: {highlight.highlight_type}")
        print(f"Context: {highlight.context}")
```

## 🎛️ Performance Optimization

### **Batch Retrieval**
```python
@dataclass
class BatchRetrievalRequest:
    """Batch retrieval for multiple queries."""
    
    queries: List[str]
    shared_config: RetrievalConfig
    per_query_filters: Optional[List[SearchFilter]] = None
    parallel_execution: bool = True
    max_concurrent: int = 5

@dataclass
class BatchRetrievalResult:
    """Batch retrieval results."""
    
    results: List[RetrievalResult]
    batch_metadata: Dict[str, Any]
    total_time: float
    average_time_per_query: float
    failed_queries: List[str] = field(default_factory=list)
```

### **Streaming Results**
```python
@dataclass
class StreamingRetrievalResult:
    """Streaming retrieval for large result sets."""
    
    query_id: str
    chunk_index: int
    total_chunks: int
    documents: List[DocumentResult]
    is_final: bool = False
    continuation_token: Optional[str] = None
```

## 📚 Integration Examples

### **With Query Analysis**
```python
async def integrated_retrieval(
    query_analysis: QueryAnalysis
) -> RetrievalResult:
    """Retrieve using query analysis results."""
    
    # Build retrieval config from analysis
    config = RetrievalConfig(
        performance_mode=get_performance_mode(query_analysis.intent),
        max_results=get_result_count(query_analysis.intent)
    )
    
    # Build filters from entities
    filters = SearchFilter(
        required_entities=[e.text for e in query_analysis.entities],
        custom_filters=query_analysis.search_filters
    )
    
    # Perform retrieval
    result = await intelligent_retriever.retrieve(
        query=query_analysis.processed_query,
        config=config,
        filters=filters,
        expansion_terms=query_analysis.expansion.related_concepts
    )
    
    return result
```

### **With RAG Orchestrator**
```python
def build_response_context(result: RetrievalResult) -> ResponseContext:
    """Build response context from retrieval results."""
    
    return ResponseContext(
        primary_sources=[
            doc for doc in result.documents[:3]
            if doc.relevance_score > 0.8
        ],
        supporting_sources=[
            doc for doc in result.documents[3:]
            if doc.relevance_score > 0.6
        ],
        highlights=[
            highlight for doc in result.documents
            for highlight in doc.highlights
        ],
        retrieval_confidence=result.confidence_score,
        processing_time=result.retrieval_time
    )
```

## 📚 Related Documentation

- **[Intelligent Retriever Service](../services/intelligent_retriever.md)** - Service implementation
- **[Query Models](query_models.md)** - Query analysis data structures
- **[RAG Orchestrator](../services/rag_orchestrator.md)** - Main pipeline integration
- **[Vector Store](../storage/vector_store.md)** - Storage layer integration

---

**The retrieval models provide comprehensive, type-safe data structures for advanced document retrieval and ranking in the Modern RAG Application.**
