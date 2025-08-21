# Query Models Documentation

## 📋 Overview

The query models module defines Pydantic data structures for query analysis, intent classification, entity extraction, and query enhancement. These models ensure type safety and data validation throughout the query processing pipeline.

## 🏗️ Core Models

### **QueryAnalysis**
Main result object containing complete query analysis:

```python
@dataclass
class QueryAnalysis:
    """Complete query analysis result."""
    
    # Query identification
    query_id: str
    original_query: str
    processed_query: str
    
    # Intent analysis
    intent: QueryIntent
    question_type: QuestionType
    confidence_score: float
    
    # Entities and concepts
    entities: List[QueryEntity]
    key_concepts: List[str]
    
    # Query enhancement
    expansion: QueryExpansion
    related_queries: List[str]
    
    # Language information
    language: LanguageInfo
    
    # Retrieval optimization
    retrieval_strategy: RetrievalStrategy
    search_filters: Dict[str, Any]
    ranking_hints: Dict[str, float]
    
    # Processing metadata
    processing_metadata: ProcessingMetadata
    timestamp: datetime = field(default_factory=datetime.now)
```

### **QueryIntent Enumeration**
Classification of user query intentions:

```python
class QueryIntent(str, Enum):
    """Query intent categories for retrieval optimization."""
    
    FACTUAL = "factual"           # Seeking specific facts or data
    ANALYTICAL = "analytical"     # Requesting analysis or explanation
    COMPARATIVE = "comparative"   # Comparing multiple items
    PROCEDURAL = "procedural"     # How-to or step-by-step information
    VERIFICATION = "verification" # Confirming or validating information
    EXPLORATORY = "exploratory"   # Open-ended research or discovery
```

### **QueryEntity**
Individual entity extracted from query:

```python
@dataclass
class QueryEntity:
    """Extracted entity from query text."""
    
    text: str                     # Entity text as it appears in query
    entity_type: EntityType       # Classification of entity
    confidence: float             # Extraction confidence (0.0-1.0)
    start_pos: int               # Character position in query
    end_pos: int                 # End character position
    context: str                 # Surrounding context
    domain_specific: bool = False # Whether entity is domain-specific
    aliases: List[str] = field(default_factory=list)  # Alternative names
```

### **QueryExpansion**
Query enhancement and expansion data:

```python
@dataclass
class QueryExpansion:
    """Query expansion and enhancement results."""
    
    original_terms: List[str]                    # Key terms from original query
    synonyms: Dict[str, List[str]]              # Term synonyms (max 3 each)
    related_concepts: List[str]                 # Contextually related terms
    domain_terms: List[str]                     # Domain-specific terminology
    temporal_context: List[str]                 # Time-related expansions
    suggested_filters: Dict[str, Any]           # Retrieval optimization filters
    expansion_strategy: ExpansionStrategy       # Strategy used for expansion
    aggressiveness: float                       # Expansion aggressiveness level
```

## 🎯 Entity Types

### **EntityType Enumeration**
Comprehensive entity classification:

```python
class EntityType(str, Enum):
    """Entity type classifications."""
    
    # People and Organizations
    PERSON = "person"
    ORGANIZATION = "organization"
    DEPARTMENT = "department"
    
    # Locations
    LOCATION = "location"
    BUILDING = "building"
    ROOM = "room"
    
    # Temporal
    DATE_TIME = "date_time"
    DURATION = "duration"
    FREQUENCY = "frequency"
    
    # Documents and References
    DOCUMENT_TYPE = "document_type"
    POLICY_ID = "policy_id"
    REFERENCE_NUMBER = "reference_number"
    
    # Numeric and Financial
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    NUMERIC = "numeric"
    MEASUREMENT = "measurement"
    
    # Technical
    SYSTEM_NAME = "system_name"
    VERSION_NUMBER = "version_number"
    TECHNICAL_TERM = "technical_term"
    
    # Concepts
    CONCEPT = "concept"
    TOPIC = "topic"
    CATEGORY = "category"
```

## 📊 Processing Metadata

### **ProcessingMetadata**
Analysis processing information:

```python
@dataclass
class ProcessingMetadata:
    """Query analysis processing metadata."""
    
    mode: ProcessingMode                    # Processing mode used
    total_time: float                      # Total processing time (seconds)
    stages: Dict[str, float]               # Time per processing stage
    nlp_used: bool                         # Whether NLP libraries were used
    fallback_used: bool                    # Whether fallback methods were used
    cache_hit: bool = False                # Whether result was cached
    error_count: int = 0                   # Number of non-fatal errors
    warnings: List[str] = field(default_factory=list)  # Processing warnings
```

### **ProcessingMode Enumeration**
Query analysis processing modes:

```python
class ProcessingMode(str, Enum):
    """Query analysis processing modes."""
    
    FAST = "fast"                 # Pattern-based only, < 50ms
    BALANCED = "balanced"         # Mixed NLP + patterns, < 200ms (DEFAULT)
    COMPREHENSIVE = "comprehensive"  # Full NLP analysis, < 500ms
```

## 🌍 Language Support

### **LanguageInfo**
Language detection and processing information:

```python
@dataclass
class LanguageInfo:
    """Language detection and processing information."""
    
    detected: str                          # ISO language code (e.g., 'en')
    confidence: float                      # Detection confidence (0.0-1.0)
    supported: bool                        # Whether language is supported
    fallback_language: str = "en"          # Fallback language if unsupported
    processing_notes: str = ""             # Additional processing information
```

## 🔍 Retrieval Strategy

### **RetrievalStrategy Enumeration**
Suggested retrieval approaches based on query analysis:

```python
class RetrievalStrategy(str, Enum):
    """Retrieval strategy recommendations."""
    
    SEMANTIC_SEARCH = "semantic_search"      # Standard semantic similarity
    ENTITY_FOCUSED = "entity_focused"        # Focus on extracted entities
    MULTI_MODAL = "multi_modal"             # Text + tables + entities
    TEMPORAL_AWARE = "temporal_aware"        # Time-sensitive retrieval
    STRUCTURE_AWARE = "structure_aware"      # Document structure-based
    COMPARATIVE = "comparative"              # Multi-document comparison
    EXPLORATORY = "exploratory"              # Broad discovery search
```

## ⚙️ Configuration Models

### **QueryAnalyzerConfig**
Configuration for query analysis behavior:

```python
@dataclass
class QueryAnalyzerConfig:
    """Configuration for QueryAnalyzer service."""
    
    # Processing settings
    processing_mode: ProcessingMode = ProcessingMode.BALANCED
    max_processing_time: float = 0.5
    
    # Entity extraction
    enable_entity_extraction: bool = True
    min_entity_confidence: float = 0.7
    max_entities_per_query: int = 20
    
    # Query expansion
    enable_query_expansion: bool = True
    expansion_aggressiveness: float = 0.3  # Conservative (0.0-1.0)
    max_expansion_terms: int = 5
    
    # Language processing
    supported_languages: List[str] = field(default_factory=lambda: ['en', 'es', 'fr'])
    default_language: str = "en"
    enable_language_detection: bool = True
    
    # Integration
    use_document_context: bool = True
    integrate_with_document_analyzer: bool = True
    
    # Caching
    cache_config: Optional[QueryCache] = None
    
    # Custom patterns (optional)
    custom_intent_patterns: Dict[str, List[str]] = field(default_factory=dict)
    custom_entity_patterns: Dict[str, List[str]] = field(default_factory=dict)
```

### **QueryCache**
Caching configuration:

```python
@dataclass
class QueryCache:
    """Query analysis caching configuration."""
    
    enabled: bool = False
    max_size: int = 1000
    ttl: int = 3600  # Time to live in seconds
    cache_strategy: str = "lru"  # LRU, LFU, or FIFO
    hash_query_params: bool = True
    cache_expansion_lookups: bool = True
```

## 🔄 Usage Examples

### **Basic Query Analysis**
```python
from models.query_models import QueryAnalysis, QueryAnalyzerConfig

# Configure analyzer
config = QueryAnalyzerConfig(
    processing_mode=ProcessingMode.BALANCED,
    expansion_aggressiveness=0.3
)

# Analyze query (result type-hinted)
analysis: QueryAnalysis = await query_analyzer.analyze_query(
    "What is the IT budget for 2024?",
    config=config
)

# Access typed results
intent: QueryIntent = analysis.intent
entities: List[QueryEntity] = analysis.entities
strategy: RetrievalStrategy = analysis.retrieval_strategy
```

### **Entity Processing**
```python
# Process extracted entities
for entity in analysis.entities:
    print(f"Entity: {entity.text}")
    print(f"Type: {entity.entity_type}")
    print(f"Confidence: {entity.confidence:.2f}")
    print(f"Context: {entity.context}")
    print(f"Domain-specific: {entity.domain_specific}")
```

### **Query Expansion Usage**
```python
# Use expansion results
expansion = analysis.expansion

# Original terms
original_terms = expansion.original_terms

# Synonyms for each term
for term, synonyms in expansion.synonyms.items():
    print(f"{term}: {', '.join(synonyms)}")

# Related concepts
related_concepts = expansion.related_concepts

# Domain-specific terms
domain_terms = expansion.domain_terms
```

## 🎛️ Model Validation

### **Pydantic Validators**
Custom validation for query models:

```python
from pydantic import validator, root_validator

class QueryEntityValidator(BaseModel):
    """Pydantic model with validation for QueryEntity."""
    
    @validator('confidence')
    def confidence_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
    
    @validator('start_pos', 'end_pos')
    def position_non_negative(cls, v):
        if v < 0:
            raise ValueError('Position must be non-negative')
        return v
    
    @root_validator
    def end_after_start(cls, values):
        start = values.get('start_pos')
        end = values.get('end_pos')
        if start is not None and end is not None and end <= start:
            raise ValueError('end_pos must be greater than start_pos')
        return values
```

### **Type Safety**
All models provide strict type checking:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type checking for development
    analysis: QueryAnalysis = get_query_analysis()
    intent: QueryIntent = analysis.intent  # Type-safe access
    entities: List[QueryEntity] = analysis.entities
```

## 📚 Integration Examples

### **With RAG Orchestrator**
```python
async def process_query_with_analysis(query: str) -> QueryResponse:
    """Process query using typed analysis results."""
    
    # Get typed analysis
    analysis: QueryAnalysis = await query_analyzer.analyze_query(query)
    
    # Use analysis for retrieval
    retrieval_params = {
        "strategy": analysis.retrieval_strategy,
        "filters": analysis.search_filters,
        "entities": [e.text for e in analysis.entities],
        "expanded_terms": analysis.expansion.related_concepts
    }
    
    # Type-safe retrieval
    results = await intelligent_retriever.retrieve(**retrieval_params)
    return results
```

### **With Intelligent Retriever**
```python
def build_retrieval_query(analysis: QueryAnalysis) -> RetrievalQuery:
    """Build retrieval query from analysis."""
    
    return RetrievalQuery(
        original_query=analysis.original_query,
        processed_query=analysis.processed_query,
        intent=analysis.intent,
        entities=[e.text for e in analysis.entities if e.confidence > 0.8],
        expansion_terms=analysis.expansion.related_concepts[:5],
        strategy=analysis.retrieval_strategy,
        filters=analysis.search_filters
    )
```

## 📚 Related Documentation

- **[Query Analyzer Service](../services/query_analyzer.md)** - Service implementation
- **[Retrieval Models](retrieval_models.md)** - Retrieval data structures
- **[RAG Orchestrator](../services/rag_orchestrator.md)** - Main pipeline
- **[System Architecture](../architecture.md)** - Overall system design

---

**The query models provide a robust, type-safe foundation for query understanding and analysis in the Modern RAG Application.**
