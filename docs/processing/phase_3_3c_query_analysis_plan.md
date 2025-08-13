# Phase 3.3c: Advanced Query Analysis - Implementation Plan

## Overview
Phase 3.3c focuses on building intelligent query understanding capabilities that complement our enhanced document processing pipeline. This phase will analyze user queries to optimize retrieval strategy and improve RAG response quality.

## Objectives

### Primary Goals
- **Query Intent Classification**: Understand what the user is trying to accomplish
- **Entity Extraction from Queries**: Identify key entities to focus retrieval
- **Context-Aware Query Expansion**: Enhance queries with relevant terms and concepts
- **Question Type Detection**: Classify query patterns for optimized processing
- **Query Quality Assessment**: Evaluate and improve query completeness

### Integration Points
- **DocumentAnalyzer Integration**: Leverage document entity knowledge for query enhancement
- **ChunkingService Integration**: Use query analysis to guide retrieval strategy
- **VectorStore Integration**: Optimize similarity search based on query characteristics

## Core Components to Build

### 1. **QueryAnalyzer Service** (`services/query_analyzer.py`)

#### Primary Responsibilities
- **Intent Classification**: Determine user's information need (factual, analytical, comparative, etc.)
- **Entity Recognition**: Extract named entities and domain-specific terms from queries
- **Question Type Detection**: Classify query structure and complexity
- **Context Enhancement**: Suggest related terms and concepts for better retrieval
- **Query Preprocessing**: Clean and normalize query text for optimal processing

#### Key Methods to Implement

##### `analyze_query(query: str, context: Optional[Dict] = None) -> QueryAnalysis`
**Purpose**: Complete query analysis with all enhancement features
```python
async def analyze_query(
    self,
    query: str,
    context: Optional[Dict] = None,
    document_context: Optional[List[str]] = None
) -> QueryAnalysis:
    """
    Comprehensive query analysis for enhanced retrieval.
    
    Args:
        query: Raw user query
        context: Optional conversation or session context
        document_context: Optional list of relevant document types
        
    Returns:
        QueryAnalysis object with all analysis results
    """
```

##### `classify_intent(query: str) -> QueryIntent`
**Purpose**: Determine the user's information need type
```python
def classify_intent(self, query: str) -> QueryIntent:
    """
    Classify query intent for optimized retrieval strategy.
    
    Intent Types:
    - FACTUAL: Seeking specific facts or data points
    - ANALYTICAL: Requesting analysis or interpretation
    - COMPARATIVE: Comparing entities, concepts, or options
    - PROCEDURAL: Looking for how-to information or processes
    - EXPLORATORY: Open-ended research or discovery
    - VERIFICATION: Confirming or validating information
    """
```

##### `extract_query_entities(query: str) -> List[QueryEntity]`
**Purpose**: Extract entities from queries for focused retrieval
```python
def extract_query_entities(self, query: str) -> List[QueryEntity]:
    """
    Extract and classify entities from user queries.
    
    Entity Types:
    - PERSON: Names of individuals
    - ORGANIZATION: Companies, departments, agencies
    - LOCATION: Places, addresses, regions
    - DATE_TIME: Temporal references
    - DOCUMENT_TYPE: Specific document categories
    - DOMAIN_TERM: Technical or domain-specific terms
    - NUMERIC: Numbers, quantities, percentages
    """
```

##### `detect_question_type(query: str) -> QuestionType`
**Purpose**: Classify query structure for appropriate processing
```python
def detect_question_type(self, query: str) -> QuestionType:
    """
    Detect question type for optimized answer generation.
    
    Question Types:
    - WHO: Person-focused queries
    - WHAT: Definition or explanation queries
    - WHERE: Location-based queries
    - WHEN: Time-based queries
    - WHY: Causation or reasoning queries
    - HOW: Process or method queries
    - HOW_MUCH/MANY: Quantitative queries
    - YES_NO: Boolean queries
    - OPEN_ENDED: Complex analytical queries
    """
```

##### `expand_query_context(query: str, entities: List[QueryEntity]) -> QueryExpansion`
**Purpose**: Enhance query with related terms and concepts
```python
def expand_query_context(
    self, 
    query: str, 
    entities: List[QueryEntity],
    document_knowledge: Optional[DocumentStructure] = None
) -> QueryExpansion:
    """
    Expand query with related terms for improved retrieval.
    
    Expansion Types:
    - SYNONYMS: Alternative terms for key concepts
    - RELATED_TERMS: Contextually related vocabulary
    - DOMAIN_CONCEPTS: Field-specific terminology
    - ENTITY_VARIANTS: Alternative entity references
    - TEMPORAL_CONTEXT: Related time periods or dates
    """
```

### 2. **Data Models** (`models/query_models.py`)

#### Core Data Structures

##### `QueryAnalysis`
```python
@dataclass
class QueryAnalysis:
    """Complete query analysis result."""
    query_id: str
    original_query: str
    intent: QueryIntent
    question_type: QuestionType
    entities: List[QueryEntity]
    expansion: QueryExpansion
    complexity_score: float
    confidence_score: float
    suggested_strategies: List[RetrievalStrategy]
    processing_time: float
    metadata: Dict[str, Any]
```

##### `QueryEntity`
```python
@dataclass
class QueryEntity:
    """Entity extracted from query."""
    text: str
    entity_type: EntityType
    start_position: int
    end_position: int
    confidence: float
    context: str
    variants: List[str]
    domain_relevance: float
```

##### `QueryExpansion`
```python
@dataclass
class QueryExpansion:
    """Query expansion with related terms."""
    original_terms: List[str]
    synonyms: Dict[str, List[str]]
    related_concepts: List[str]
    domain_terms: List[str]
    temporal_context: List[str]
    expansion_confidence: float
    suggested_filters: Dict[str, Any]
```

### 3. **Query Enhancement Pipeline**

#### Stage 1: Query Preprocessing
```python
def preprocess_query(self, query: str) -> str:
    """
    Clean and normalize query text.
    
    Steps:
    1. Unicode normalization
    2. Spelling correction (optional)
    3. Stop word handling (context-aware)
    4. Punctuation normalization
    5. Case normalization
    """
```

#### Stage 2: Intent and Type Classification
```python
async def classify_query_characteristics(self, query: str) -> Dict[str, Any]:
    """
    Determine query intent and question type.
    
    Uses combination of:
    - Pattern matching for question words
    - Semantic analysis for intent detection
    - Context clues from query structure
    - Domain-specific indicators
    """
```

#### Stage 3: Entity Recognition and Enhancement
```python
async def enhance_query_entities(
    self, 
    query: str,
    document_context: Optional[List[DocumentStructure]] = None
) -> List[QueryEntity]:
    """
    Extract and enhance entities with document knowledge.
    
    Enhancement includes:
    - Entity normalization using document knowledge
    - Variant detection (abbreviations, full forms)
    - Domain-specific entity classification
    - Confidence scoring based on document context
    """
```

#### Stage 4: Context-Aware Expansion
```python
async def generate_query_expansion(
    self,
    query: str,
    entities: List[QueryEntity],
    intent: QueryIntent,
    document_types: Optional[List[str]] = None
) -> QueryExpansion:
    """
    Generate context-aware query expansion.
    
    Expansion strategies:
    - Synonym generation using embeddings similarity
    - Domain concept mapping from document analysis
    - Temporal context expansion for date-related queries
    - Entity-specific expansions (e.g., org names -> departments)
    """
```

## Implementation Architecture

### NLP Integration Strategy
Following our established pattern of offline-capable NLP with graceful fallbacks:

```python
class QueryAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger(__name__)
        
        # NLP Library Integration (with fallbacks)
        self.nltk_ready = self._setup_nltk()
        self.spacy_ready = self._setup_spacy()
        
        # Document knowledge integration
        self.document_analyzer = DocumentAnalyzer()
        
        # Query processing components
        self._intent_classifier = self._setup_intent_classifier()
        self._entity_extractor = self._setup_entity_extractor()
        self._query_expander = self._setup_query_expander()
    
    def _setup_nltk(self) -> bool:
        """Setup NLTK for query processing with fallback."""
        try:
            import nltk
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk.corpus import stopwords, wordnet
            from nltk.stem import WordNetLemmatizer
            
            # Download required data if not available
            required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            for data_name in required_data:
                try:
                    nltk.data.find(f'tokenizers/{data_name}')
                except LookupError:
                    nltk.download(data_name, quiet=True)
            
            self.word_tokenize = word_tokenize
            self.stopwords = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.wordnet = wordnet
            
            return True
            
        except Exception as e:
            self.logger.warning(f"NLTK setup failed: {e}, using basic tokenization")
            return False
    
    def _setup_spacy(self) -> bool:
        """Setup spaCy for advanced query processing."""
        try:
            import spacy
            
            # Try to load English model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy English model not found, using basic pipeline")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"spaCy setup failed: {e}, using regex-based processing")
            return False
```

### Intent Classification Implementation

#### Pattern-Based Classification (Fast, Offline)
```python
def classify_intent_patterns(self, query: str) -> QueryIntent:
    """Fast pattern-based intent classification."""
    
    query_lower = query.lower()
    
    # Factual indicators
    factual_patterns = [
        r'\bwhat is\b', r'\bwho is\b', r'\bwhere is\b',
        r'\bwhen did\b', r'\bwhich\b', r'\blist\b',
        r'\bshow me\b', r'\bfind\b', r'\bget\b'
    ]
    
    # Analytical indicators  
    analytical_patterns = [
        r'\bwhy\b', r'\bhow does\b', r'\banalyze\b',
        r'\bexplain\b', r'\bevaluate\b', r'\bassess\b',
        r'\bcompare\b', r'\bcontrast\b'
    ]
    
    # Procedural indicators
    procedural_patterns = [
        r'\bhow to\b', r'\bsteps\b', r'\bprocess\b',
        r'\bprocedure\b', r'\binstructions\b', r'\bguide\b'
    ]
    
    # Pattern matching with confidence scoring
    for pattern in factual_patterns:
        if re.search(pattern, query_lower):
            return QueryIntent.FACTUAL
    
    for pattern in analytical_patterns:
        if re.search(pattern, query_lower):
            return QueryIntent.ANALYTICAL
    
    for pattern in procedural_patterns:
        if re.search(pattern, query_lower):
            return QueryIntent.PROCEDURAL
    
    # Default to exploratory for complex queries
    return QueryIntent.EXPLORATORY
```

#### Semantic Classification (Enhanced with NLP)
```python
async def classify_intent_semantic(self, query: str) -> QueryIntent:
    """Enhanced semantic intent classification using NLP."""
    
    if not self.spacy_ready:
        return self.classify_intent_patterns(query)
    
    doc = self.nlp(query)
    
    # Analyze semantic features
    features = {
        'question_words': [token.text.lower() for token in doc if token.pos_ == 'ADV' and token.text.lower() in ['who', 'what', 'where', 'when', 'why', 'how']],
        'verbs': [token.lemma_ for token in doc if token.pos_ == 'VERB'],
        'entities': [(ent.text, ent.label_) for ent in doc.ents],
        'dependency_patterns': [(token.dep_, token.head.text) for token in doc]
    }
    
    # Enhanced classification logic using semantic features
    intent_scores = {
        QueryIntent.FACTUAL: 0.0,
        QueryIntent.ANALYTICAL: 0.0,
        QueryIntent.COMPARATIVE: 0.0,
        QueryIntent.PROCEDURAL: 0.0,
        QueryIntent.EXPLORATORY: 0.0,
        QueryIntent.VERIFICATION: 0.0
    }
    
    # Score based on semantic features
    # ... detailed scoring logic ...
    
    return max(intent_scores, key=intent_scores.get)
```

### Entity Extraction Implementation

#### Query-Specific Entity Extraction
```python
def extract_query_entities_enhanced(self, query: str) -> List[QueryEntity]:
    """Extract entities specifically relevant to queries."""
    
    entities = []
    
    if self.spacy_ready:
        entities.extend(self._extract_entities_spacy(query))
    
    entities.extend(self._extract_entities_patterns(query))
    
    # Merge and deduplicate
    return self._merge_query_entities(entities)

def _extract_entities_spacy(self, query: str) -> List[QueryEntity]:
    """Extract entities using spaCy."""
    doc = self.nlp(query)
    entities = []
    
    for ent in doc.ents:
        entity = QueryEntity(
            text=ent.text,
            entity_type=self._map_spacy_label_to_query_entity(ent.label_),
            start_position=ent.start_char,
            end_position=ent.end_char,
            confidence=0.9,  # spaCy is generally high confidence
            context=query[max(0, ent.start_char-20):ent.end_char+20],
            variants=[],
            domain_relevance=0.8
        )
        entities.append(entity)
    
    return entities

def _extract_entities_patterns(self, query: str) -> List[QueryEntity]:
    """Extract entities using regex patterns."""
    entities = []
    
    # Date patterns
    date_patterns = [
        (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', 'DATE'),
        (r'\b\d{4}-\d{1,2}-\d{1,2}\b', 'DATE'),
        (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b', 'DATE'),
        (r'\b(last|this|next)\s+(week|month|year|quarter)\b', 'RELATIVE_DATE')
    ]
    
    # Document type patterns
    doc_type_patterns = [
        (r'\b(policy|memo|report|manual|guide|document|form|notice|circular)\b', 'DOCUMENT_TYPE'),
        (r'\b(pdf|doc|docx|txt)\b', 'FILE_TYPE')
    ]
    
    # Numeric patterns
    numeric_patterns = [
        (r'\b\d+%\b', 'PERCENTAGE'),
        (r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b', 'CURRENCY'),
        (r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', 'NUMBER')
    ]
    
    all_patterns = date_patterns + doc_type_patterns + numeric_patterns
    
    for pattern, entity_type in all_patterns:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            entity = QueryEntity(
                text=match.group(),
                entity_type=entity_type,
                start_position=match.start(),
                end_position=match.end(),
                confidence=0.8,
                context=query[max(0, match.start()-15):match.end()+15],
                variants=[],
                domain_relevance=0.7
            )
            entities.append(entity)
    
    return entities
```

### Query Expansion Implementation

#### Semantic Expansion Strategy
```python
async def expand_query_semantic(
    self, 
    query: str, 
    entities: List[QueryEntity],
    document_context: Optional[List[DocumentStructure]] = None
) -> QueryExpansion:
    """Generate semantic query expansion."""
    
    original_terms = self._extract_key_terms(query)
    
    expansion = QueryExpansion(
        original_terms=original_terms,
        synonyms={},
        related_concepts=[],
        domain_terms=[],
        temporal_context=[],
        expansion_confidence=0.0,
        suggested_filters={}
    )
    
    # Generate synonyms
    if self.nltk_ready:
        expansion.synonyms = self._generate_synonyms_wordnet(original_terms)
    
    # Domain-specific expansion using document knowledge
    if document_context:
        expansion.domain_terms = self._extract_domain_terms(query, document_context)
        expansion.related_concepts = self._find_related_concepts(entities, document_context)
    
    # Temporal expansion
    expansion.temporal_context = self._expand_temporal_context(entities)
    
    # Calculate overall expansion confidence
    expansion.expansion_confidence = self._calculate_expansion_confidence(expansion)
    
    return expansion

def _generate_synonyms_wordnet(self, terms: List[str]) -> Dict[str, List[str]]:
    """Generate synonyms using WordNet."""
    synonyms = {}
    
    for term in terms:
        term_synonyms = []
        
        # Get all synsets for the term
        synsets = self.wordnet.synsets(term.lower())
        
        for synset in synsets[:3]:  # Limit to top 3 synsets
            for lemma in synset.lemmas()[:5]:  # Limit to top 5 lemmas per synset
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != term.lower() and synonym not in term_synonyms:
                    term_synonyms.append(synonym)
        
        if term_synonyms:
            synonyms[term] = term_synonyms[:5]  # Limit to top 5 synonyms
    
    return synonyms

def _extract_domain_terms(
    self, 
    query: str, 
    document_context: List[DocumentStructure]
) -> List[str]:
    """Extract domain-specific terms from document context."""
    domain_terms = []
    
    # Analyze entities across document context
    all_entities = []
    for doc_structure in document_context:
        all_entities.extend(doc_structure.entities)
    
    # Find entities that might be relevant to the query
    query_lower = query.lower()
    for entity in all_entities:
        # Simple relevance check - can be enhanced with embedding similarity
        if any(word in entity.text.lower() for word in query_lower.split()):
            domain_terms.append(entity.text)
    
    return list(set(domain_terms))  # Remove duplicates
```

## Integration with Existing Components

### Integration with DocumentAnalyzer
```python
class QueryDocumentMatcher:
    """Match query characteristics with document structure knowledge."""
    
    def __init__(self, document_analyzer: DocumentAnalyzer):
        self.document_analyzer = document_analyzer
    
    def suggest_retrieval_strategy(
        self, 
        query_analysis: QueryAnalysis,
        available_documents: List[DocumentStructure]
    ) -> RetrievalStrategy:
        """Suggest optimal retrieval strategy based on query and documents."""
        
        strategy = RetrievalStrategy()
        
        # Intent-based strategy selection
        if query_analysis.intent == QueryIntent.FACTUAL:
            strategy.focus_on_entities = True
            strategy.prefer_tables = True
            
        elif query_analysis.intent == QueryIntent.ANALYTICAL:
            strategy.include_context = True
            strategy.expand_sections = True
            
        elif query_analysis.intent == QueryIntent.COMPARATIVE:
            strategy.multi_document = True
            strategy.preserve_relationships = True
        
        # Entity-based document filtering
        strategy.document_type_filters = self._suggest_document_filters(
            query_analysis.entities, available_documents
        )
        
        return strategy
```

### Integration with ChunkingService
```python
def enhance_retrieval_with_query_analysis(
    self,
    query_analysis: QueryAnalysis,
    chunks: List[TextChunk]
) -> List[TextChunk]:
    """Re-rank and filter chunks based on query analysis."""
    
    enhanced_chunks = []
    
    for chunk in chunks:
        # Calculate query-chunk relevance score
        relevance_score = self._calculate_query_chunk_relevance(
            query_analysis, chunk
        )
        
        # Enhance chunk metadata with query-specific information
        enhanced_metadata = {
            **chunk.metadata,
            'query_relevance_score': relevance_score,
            'matches_intent': self._matches_query_intent(query_analysis.intent, chunk),
            'contains_query_entities': self._contains_query_entities(query_analysis.entities, chunk)
        }
        
        chunk.metadata = enhanced_metadata
        enhanced_chunks.append(chunk)
    
    # Sort by relevance score
    enhanced_chunks.sort(key=lambda c: c.metadata['query_relevance_score'], reverse=True)
    
    return enhanced_chunks
```

## Testing Strategy

### Unit Tests
```python
# tests/test_services/test_query_analyzer.py

class TestQueryAnalyzer:
    
    async def test_intent_classification(self):
        """Test query intent classification accuracy."""
        test_cases = [
            ("What is the capital of France?", QueryIntent.FACTUAL),
            ("Why did the policy change?", QueryIntent.ANALYTICAL), 
            ("How do I submit a form?", QueryIntent.PROCEDURAL),
            ("Compare budget 2023 vs 2024", QueryIntent.COMPARATIVE)
        ]
        
        analyzer = QueryAnalyzer()
        
        for query, expected_intent in test_cases:
            result = await analyzer.analyze_query(query)
            assert result.intent == expected_intent
    
    async def test_entity_extraction(self):
        """Test entity extraction from queries."""
        query = "Find documents from Department of Health dated January 2024"
        
        analyzer = QueryAnalyzer()
        result = await analyzer.analyze_query(query)
        
        # Should extract department and date entities
        entity_types = [e.entity_type for e in result.entities]
        assert 'ORGANIZATION' in entity_types
        assert 'DATE' in entity_types
    
    async def test_query_expansion(self):
        """Test query expansion functionality."""
        query = "budget allocation report"
        
        analyzer = QueryAnalyzer()
        result = await analyzer.analyze_query(query)
        
        # Should have synonyms and related terms
        assert len(result.expansion.synonyms) > 0
        assert len(result.expansion.related_concepts) > 0
```

## Performance Characteristics

### Expected Performance
- **Query Analysis Time**: < 100ms for simple queries, < 500ms for complex queries
- **Memory Usage**: < 50MB base overhead
- **Accuracy Targets**:
  - Intent classification: > 85%
  - Entity extraction: > 80%
  - Question type detection: > 90%

### Scalability Considerations
- **Batch Processing**: Support for analyzing multiple queries simultaneously
- **Caching**: Cache analysis results for repeated queries
- **Offline Operation**: All processing works without internet connectivity

## Implementation Timeline

### Week 1: Core Infrastructure
- Set up QueryAnalyzer service structure
- Implement basic intent classification
- Create data models and enums

### Week 2: Entity Extraction & Question Type Detection
- Implement entity extraction with NLP integration
- Build question type classification
- Add pattern-based fallbacks

### Week 3: Query Expansion & Integration
- Implement query expansion strategies
- Integrate with DocumentAnalyzer
- Build retrieval strategy suggestions

### Week 4: Testing & Optimization
- Comprehensive test suite
- Performance optimization
- Documentation completion

## Success Metrics

### Functional Success
- ✅ **Intent Classification**: 85%+ accuracy on test queries
- ✅ **Entity Extraction**: 80%+ accuracy for standard entity types
- ✅ **Question Type Detection**: 90%+ accuracy for common patterns
- ✅ **Query Expansion**: Meaningful related terms for 70%+ of queries

### Integration Success  
- ✅ **DocumentAnalyzer Integration**: Seamless knowledge transfer
- ✅ **ChunkingService Enhancement**: Improved retrieval relevance
- ✅ **Performance**: Sub-second query analysis
- ✅ **Fallback Reliability**: 100% uptime even without NLP libraries

This completes the foundation for advanced query understanding that will enable **Phase 3.3d: Intelligent Retrieval** to deliver optimal results!
