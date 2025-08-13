"""
QueryAnalyzer Service - Advanced Query Understanding for Modern RAG

This service provides intelligent query analysis for optimized document retrieval.
It combines pattern-based processing with advanced NLP capabilities while maintaining
offline operation and graceful fallbacks.

Key Features:
- Query intent classification (6 types: factual, analytical, comparative, etc.)
- Entity extraction with domain-specific recognition
- Conservative query expansion for precision
- Integration with DocumentAnalyzer knowledge
- Configurable performance modes (FAST/BALANCED/COMPREHENSIVE)
- Optional caching for repeated queries
- 100% offline operation after setup

Architecture:
- Offline-first design with NLTK, spaCy integration
- Pattern-based fallbacks ensuring 100% uptime
- Conservative expansion strategy for precision over recall
- Integration points with existing pipeline components
"""

import asyncio
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import hashlib

# Core dependencies
from config.settings import Settings
from config.logging_config import get_logger

# Data models
from models.query_models import (
    QueryAnalysis, QueryEntity, QueryExpansion, QueryIntent, 
    QuestionType, EntityType, ProcessingMode, RetrievalStrategy,
    QueryAnalyzerConfig, QueryCache
)

# Integration with existing services
from services.document_analyzer import DocumentAnalyzer, DocumentStructure

# NLP Libraries (with graceful fallbacks)
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import wordnet, stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class QueryAnalyzer:
    """
    Advanced query analysis service for intelligent retrieval optimization.
    
    **Processing Modes**:
    - FAST: Pattern-based only, < 50ms
    - BALANCED: Mixed NLP + patterns, < 200ms (DEFAULT)
    - COMPREHENSIVE: Full NLP analysis, < 500ms
    
    **Usage Example**:
    ```python
    # Basic usage with default config
    analyzer = QueryAnalyzer()
    analysis = await analyzer.analyze_query("What is the budget for IT department?")
    
    # Advanced usage with custom config
    config = QueryAnalyzerConfig(
        processing_mode=ProcessingMode.COMPREHENSIVE,
        expansion_aggressiveness=0.3,  # Conservative
        cache_config=QueryCache(enabled=True)
    )
    analyzer = QueryAnalyzer(config=config)
    analysis = await analyzer.analyze_query(query, document_context=doc_structures)
    ```
    
    **Integration Points**:
    - DocumentAnalyzer: Leverage entity knowledge for query enhancement
    - ChunkingService: Provide retrieval strategy hints
    - EmbeddingService: Optional embedding similarity (when necessary)
    - VectorStore: Pass analysis for optimized search
    """

    def __init__(
        self, 
        settings: Optional[Settings] = None,
        config: Optional[QueryAnalyzerConfig] = None
    ):
        """Initialize QueryAnalyzer with configuration."""
        self.settings = settings or Settings()
        self.config = config or QueryAnalyzerConfig()
        self.logger = get_logger(__name__)
        
        # NLP setup with fallbacks
        self.nltk_ready = self._setup_nltk()
        self.spacy_ready = self._setup_spacy()
        self.nlp = None
        
        # Document knowledge integration
        self.document_analyzer = None
        if self.config.integrate_document_knowledge:
            try:
                self.document_analyzer = DocumentAnalyzer(settings=self.settings)
                self.logger.info("✅ DocumentAnalyzer integration enabled")
            except Exception as e:
                self.logger.warning(f"⚠️ DocumentAnalyzer integration failed: {e}")
        
        # Initialize processing components
        self._intent_patterns = self._load_intent_patterns()
        self._entity_patterns = self._load_entity_patterns()
        self._question_patterns = self._load_question_patterns()
        
        # Optional caching
        self._cache: Dict[str, QueryAnalysis] = {}
        self._expansion_cache: Dict[str, List[str]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Performance tracking
        self._analysis_count = 0
        self._total_processing_time = 0.0
        
        self.logger.info(f"✅ QueryAnalyzer initialized - Mode: {self.config.processing_mode.value}")
        self.logger.info(f"🔧 NLP Status: NLTK={self.nltk_ready}, spaCy={self.spacy_ready}")

    def _setup_nltk(self) -> bool:
        """Setup NLTK with required data packages."""
        if not NLTK_AVAILABLE:
            self.logger.warning("⚠️ NLTK not available - using pattern fallbacks")
            return False
        
        try:
            # Check for required NLTK data
            required_data = [
                'tokenizers/punkt',
                'corpora/wordnet',
                'corpora/stopwords',
                'taggers/averaged_perceptron_tagger'
            ]
            
            missing_data = []
            for data_path in required_data:
                try:
                    nltk.data.find(data_path)
                except LookupError:
                    missing_data.append(data_path)
            
            if missing_data:
                self.logger.warning(f"⚠️ NLTK missing data: {missing_data}")
                self.logger.info("📥 Attempting to download missing NLTK data...")
                
                for data_path in missing_data:
                    package_name = data_path.split('/')[-1]
                    try:
                        nltk.download(package_name, quiet=True)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to download {package_name}: {e}")
                        return False
            
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            self.logger.info("✅ NLTK initialized successfully")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ NLTK setup failed: {e}")
            return False

    def _setup_spacy(self) -> bool:
        """Setup spaCy with English language model."""
        if not SPACY_AVAILABLE:
            self.logger.warning("⚠️ spaCy not available - using pattern fallbacks")
            return False
        
        try:
            # Try to load English model (prefer en_core_web_sm)
            model_names = ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
            
            for model_name in model_names:
                try:
                    self.nlp = spacy.load(model_name)
                    self.logger.info(f"✅ spaCy loaded with {model_name}")
                    return True
                except OSError:
                    continue
            
            # If no pre-trained model available, try blank English
            self.nlp = spacy.blank('en')
            self.logger.warning("⚠️ Using blank spaCy model - entity recognition limited")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ spaCy setup failed: {e}")
            return False

    def _load_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Load pattern-based intent classification rules."""
        patterns = {
            QueryIntent.FACTUAL: [
                r'\bwhat\s+is\b', r'\bwho\s+is\b', r'\bwhere\s+is\b', r'\bwhen\s+is\b',
                r'\bhow\s+much\b', r'\bhow\s+many\b', r'\bwhich\b', r'\blist\b',
                r'\bshow\s+me\b', r'\btell\s+me\b', r'\bfind\b', r'\bget\b'
            ],
            QueryIntent.ANALYTICAL: [
                r'\bwhy\b', r'\bhow\s+come\b', r'\bexplain\b', r'\banalyze\b',
                r'\bwhat\s+caused\b', r'\bwhat\s+led\s+to\b', r'\breason\b',
                r'\bimpact\b', r'\beffect\b', r'\bconsequence\b', r'\bresult\b'
            ],
            QueryIntent.COMPARATIVE: [
                r'\bcompare\b', r'\bcontrast\b', r'\bdifference\b', r'\bsimilar\b',
                r'\bvs\b', r'\bversus\b', r'\bbetter\b', r'\bworse\b', r'\bhigher\b',
                r'\blower\b', r'\bmore\s+than\b', r'\bless\s+than\b', r'\bbetween\b'
            ],
            QueryIntent.PROCEDURAL: [
                r'\bhow\s+to\b', r'\bhow\s+do\s+i\b', r'\bsteps\b', r'\bprocess\b',
                r'\bprocedure\b', r'\binstructions\b', r'\bguide\b', r'\bmethod\b',
                r'\bway\s+to\b', r'\bapproach\b', r'\bsubmit\b', r'\bapply\b'
            ],
            QueryIntent.VERIFICATION: [
                r'\bis\s+this\b', r'\bis\s+it\b', r'\bconfirm\b', r'\bverify\b',
                r'\bcheck\b', r'\bvalid\b', r'\bcorrect\b', r'\btrue\b',
                r'\bstill\s+active\b', r'\bstill\s+valid\b', r'\bup\s+to\s+date\b'
            ],
            QueryIntent.EXPLORATORY: [
                r'\btell\s+me\s+about\b', r'\blearn\s+about\b', r'\bexplore\b',
                r'\boverview\b', r'\bsummary\b', r'\bgeneral\b', r'\bbroad\b',
                r'\bwhat\s+do\s+you\s+know\b', r'\banything\s+about\b'
            ]
        }
        
        # Add custom patterns from config
        for intent_str, custom_patterns in self.config.custom_intent_patterns.items():
            try:
                intent = QueryIntent(intent_str)
                if intent in patterns:
                    patterns[intent].extend(custom_patterns)
                else:
                    patterns[intent] = custom_patterns
            except ValueError:
                self.logger.warning(f"⚠️ Invalid custom intent: {intent_str}")
        
        return patterns

    def _load_entity_patterns(self) -> Dict[EntityType, List[str]]:
        """Load pattern-based entity extraction rules."""
        patterns = {
            EntityType.DATE_TIME: [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',  # MM/DD/YYYY or MM-DD-YYYY
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD or YYYY-MM-DD
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b',
                r'\bQ[1-4]\s+\d{4}\b',  # Q1 2024
                r'\b\d{4}\s+Q[1-4]\b',  # 2024 Q1
                r'\b(?:last|this|next)\s+(?:year|month|week|quarter)\b',
                r'\b\d{4}\b(?=\s*(?:budget|report|data))'  # Year before certain keywords
            ],
            EntityType.CURRENCY: [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',  # $1,000.00
                r'\b\d+(?:,\d{3})*\s*(?:dollars|USD|cents)\b',
                r'\b\d+(?:\.\d+)?\s*(?:million|billion|thousand)\s*(?:dollars|USD)?\b'
            ],
            EntityType.NUMERIC: [
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent)\b',  # Percentages
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',  # General numbers
                r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b'  # Written numbers
            ],
            EntityType.DOCUMENT_TYPE: [
                r'\b(?:report|policy|memo|memorandum|form|document|manual|guide|circular|notice|bulletin)\b',
                r'\b(?:budget|financial|quarterly|annual|monthly|weekly|daily)\s+(?:report|statement|summary)\b',
                r'\b(?:change|request|CR|incident|ticket|issue)\s*(?:\d+|#\d+)?\b'
            ],
            EntityType.ORGANIZATION: [
                r'\b(?:Department|Dept|Ministry|Office|Bureau|Agency|Division|Unit)\s+of\s+\w+(?:\s+\w+)*\b',
                r'\b(?:IT|HR|Finance|Marketing|Sales|Operations|Legal|Compliance|Security)\s+(?:Department|Dept|Team|Division|Unit)\b',
                r'\b[A-Z]{2,5}(?:\s+[A-Z]{2,5})*\b(?=\s*(?:department|team|unit|division))',  # Acronyms
            ],
            EntityType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            EntityType.PHONE: [
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
            ],
            EntityType.POLICY_ID: [
                r'\b[A-Z]{2,5}[-_]?\d{4}[-_]?\d{3,4}\b',  # Policy format like HHS-2024-001
                r'\bPolicy\s+(?:No\.?|Number)\s*[:.]?\s*[A-Z0-9-]{3,}\b'
            ]
        }
        
        # Add custom patterns from config
        for entity_str, custom_patterns in self.config.custom_entity_patterns.items():
            try:
                entity_type = EntityType(entity_str)
                if entity_type in patterns:
                    patterns[entity_type].extend(custom_patterns)
                else:
                    patterns[entity_type] = custom_patterns
            except ValueError:
                self.logger.warning(f"⚠️ Invalid custom entity type: {entity_str}")
        
        return patterns

    def _load_question_patterns(self) -> Dict[QuestionType, List[str]]:
        """Load question type classification patterns."""
        return {
            QuestionType.WHO: [r'\bwho\b'],
            QuestionType.WHAT: [r'\bwhat\b'],
            QuestionType.WHERE: [r'\bwhere\b'],
            QuestionType.WHEN: [r'\bwhen\b'],
            QuestionType.WHY: [r'\bwhy\b', r'\bhow\s+come\b'],
            QuestionType.HOW: [r'\bhow\b(?!\s+(?:much|many))'],
            QuestionType.HOW_MUCH: [r'\bhow\s+much\b'],
            QuestionType.HOW_MANY: [r'\bhow\s+many\b'],
            QuestionType.YES_NO: [
                r'\b(?:is|are|was|were|do|does|did|can|could|will|would|should|shall)\b.*\?',
                r'\b(?:true|false|correct|valid)\b'
            ],
            QuestionType.OPEN_ENDED: [r'.*']  # Fallback pattern
        }

    def _get_cache_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Generate cache key for query and context."""
        if not self.config.cache_config or not self.config.cache_config.enabled:
            return ""
        
        # Create hash from query and relevant context
        cache_data = {"query": query.lower().strip()}
        if context:
            # Only include stable context elements
            stable_keys = ['document_types', 'user_id', 'session_id']
            cache_data.update({k: v for k, v in context.items() if k in stable_keys})
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if not self.config.cache_config or cache_key not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[cache_key]
        ttl = timedelta(seconds=self.config.cache_config.ttl_seconds)
        return datetime.now() - cache_time < ttl

    def _clean_cache(self):
        """Remove expired cache entries."""
        if not self.config.cache_config:
            return
        
        current_time = datetime.now()
        ttl = timedelta(seconds=self.config.cache_config.ttl_seconds)
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > ttl
        ]
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
            self._expansion_cache.pop(key, None)

    async def analyze_query(
        self,
        query: str,
        context: Optional[Dict] = None,
        document_context: Optional[List[DocumentStructure]] = None
    ) -> QueryAnalysis:
        """
        Perform comprehensive query analysis for intelligent retrieval.
        
        Args:
            query: Raw user query text
            context: Optional conversation or session context
            document_context: Optional document structures for knowledge integration
            
        Returns:
            QueryAnalysis with all enhancement features
            
        **Usage Example**:
        ```python
        analysis = await analyzer.analyze_query(
            "What was the IT budget allocation in Q2 2024?",
            context={"session_id": "12345"},
            document_context=document_structures
        )
        
        print(f"Intent: {analysis.intent}")
        print(f"Entities: {[e.text for e in analysis.entities]}")
        print(f"Expansion: {analysis.expansion.synonyms}")
        ```
        """
        start_time = time.time()
        
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query = query.strip()
        
        # Check cache first
        cache_key = self._get_cache_key(query, context)
        if cache_key and self._is_cache_valid(cache_key):
            cached_result = self._cache.get(cache_key)
            if cached_result:
                self.logger.debug(f"📋 Cache hit for query: {query[:50]}...")
                return cached_result
        
        # Clean expired cache entries periodically
        if self._analysis_count % 100 == 0:
            self._clean_cache()
        
        try:
            # Stage 1: Query preprocessing
            processed_query = self._preprocess_query(query)
            
            # Stage 2: Intent classification
            intent = await self._classify_intent(processed_query)
            
            # Stage 3: Question type detection
            question_type = self._detect_question_type(processed_query)
            
            # Stage 4: Entity extraction
            entities = await self._extract_entities(processed_query)
            
            # Stage 5: Query expansion (conservative approach)
            expansion = await self._expand_query(
                processed_query, entities, document_context
            )
            
            # Stage 6: Complexity and confidence scoring
            complexity_score = self._calculate_complexity(processed_query, entities)
            confidence_score = self._calculate_confidence(intent, entities, expansion)
            
            # Stage 7: Retrieval strategy suggestions
            suggested_strategies = self._suggest_retrieval_strategies(
                intent, question_type, entities
            )
            
            # Create comprehensive analysis result
            analysis = QueryAnalysis(
                query_id="",  # Will be generated in __post_init__
                original_query=query,
                processed_query=processed_query,
                intent=intent,
                question_type=question_type,
                entities=entities,
                expansion=expansion,
                complexity_score=complexity_score,
                confidence_score=confidence_score,
                suggested_strategies=suggested_strategies,
                processing_time=round((time.time() - start_time) * 1000, 2),
                processing_mode=self.config.processing_mode,
                metadata={
                    'nlp_libraries_used': {
                        'nltk': self.nltk_ready,
                        'spacy': self.spacy_ready
                    },
                    'context_provided': context is not None,
                    'document_context_count': len(document_context) if document_context else 0,
                    'fallback_patterns_used': not (self.nltk_ready and self.spacy_ready)
                }
            )
            
            # Cache result if caching enabled
            if cache_key and self.config.cache_config and self.config.cache_config.enabled:
                self._cache[cache_key] = analysis
                self._cache_timestamps[cache_key] = datetime.now()
                
                # Enforce cache size limit
                if len(self._cache) > self.config.cache_config.max_entries:
                    oldest_key = min(self._cache_timestamps.items(), key=lambda x: x[1])[0]
                    self._cache.pop(oldest_key, None)
                    self._cache_timestamps.pop(oldest_key, None)
            
            # Update performance tracking
            self._analysis_count += 1
            self._total_processing_time += analysis.processing_time
            
            self.logger.debug(
                f"📊 Query analyzed: {query[:50]}... "
                f"(Intent: {intent.value}, Entities: {len(entities)}, "
                f"Time: {analysis.processing_time}ms)"
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Query analysis failed: {e}", exc_info=True)
            
            # Return minimal analysis on error
            fallback_analysis = QueryAnalysis(
                query_id="",
                original_query=query,
                processed_query=query,
                intent=QueryIntent.EXPLORATORY,
                question_type=QuestionType.OPEN_ENDED,
                entities=[],
                expansion=QueryExpansion(original_terms=query.split()),
                complexity_score=0.5,
                confidence_score=0.1,
                suggested_strategies=[RetrievalStrategy.SEMANTIC_SEARCH],
                processing_time=round((time.time() - start_time) * 1000, 2),
                processing_mode=ProcessingMode.FAST,
                metadata={'error': str(e), 'fallback_used': True}
            )
            
            return fallback_analysis

    def _preprocess_query(self, query: str) -> str:
        """Clean and normalize query text for processing."""
        # Unicode normalization
        import unicodedata
        query = unicodedata.normalize('NFKD', query)
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Fix common punctuation issues
        query = re.sub(r'\s*([?.!])\s*$', r'\1', query)
        
        return query

    async def _classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent using configured processing mode."""
        if self.config.processing_mode == ProcessingMode.FAST:
            return self._classify_intent_patterns(query)
        elif self.config.processing_mode == ProcessingMode.COMPREHENSIVE and self.spacy_ready:
            return await self._classify_intent_semantic(query)
        else:  # BALANCED or fallback
            return self._classify_intent_patterns(query)

    def _classify_intent_patterns(self, query: str) -> QueryIntent:
        """Fast pattern-based intent classification."""
        query_lower = query.lower()
        
        # Score each intent based on pattern matches
        intent_scores = defaultdict(float)
        
        for intent, patterns in self._intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent_scores[intent] += 1.0
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        # Default fallback based on query characteristics
        if '?' in query:
            return QueryIntent.FACTUAL
        elif any(word in query_lower for word in ['compare', 'vs', 'difference']):
            return QueryIntent.COMPARATIVE
        elif any(word in query_lower for word in ['how', 'step', 'process']):
            return QueryIntent.PROCEDURAL
        else:
            return QueryIntent.EXPLORATORY

    async def _classify_intent_semantic(self, query: str) -> QueryIntent:
        """Advanced semantic intent classification using spaCy."""
        if not self.spacy_ready:
            return self._classify_intent_patterns(query)
        
        try:
            doc = self.nlp(query)
            
            # Analyze semantic features
            features = {
                'question_words': [token.text.lower() for token in doc 
                                 if token.pos_ == 'ADV' and token.text.lower() in 
                                 ['who', 'what', 'where', 'when', 'why', 'how']],
                'verbs': [token.lemma_ for token in doc if token.pos_ == 'VERB'],
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'root_verb': doc[0].head.lemma_ if doc else None
            }
            
            # Enhanced scoring using semantic features
            intent_scores = defaultdict(float)
            
            # Analyze question words and verb patterns
            for qword in features['question_words']:
                if qword in ['what', 'who', 'where', 'when']:
                    intent_scores[QueryIntent.FACTUAL] += 0.8
                elif qword == 'why':
                    intent_scores[QueryIntent.ANALYTICAL] += 0.9
                elif qword == 'how':
                    intent_scores[QueryIntent.PROCEDURAL] += 0.8
            
            # Analyze root verb semantics
            analytical_verbs = {'analyze', 'explain', 'cause', 'result', 'impact'}
            comparative_verbs = {'compare', 'contrast', 'differ', 'similar'}
            procedural_verbs = {'submit', 'apply', 'process', 'perform', 'execute'}
            
            for verb in features['verbs']:
                if verb in analytical_verbs:
                    intent_scores[QueryIntent.ANALYTICAL] += 0.7
                elif verb in comparative_verbs:
                    intent_scores[QueryIntent.COMPARATIVE] += 0.8
                elif verb in procedural_verbs:
                    intent_scores[QueryIntent.PROCEDURAL] += 0.7
            
            # Pattern-based scoring as fallback
            pattern_intent = self._classify_intent_patterns(query)
            intent_scores[pattern_intent] += 0.5
            
            if intent_scores:
                return max(intent_scores, key=intent_scores.get)
            else:
                return pattern_intent
                
        except Exception as e:
            self.logger.warning(f"⚠️ Semantic intent classification failed: {e}")
            return self._classify_intent_patterns(query)

    def _detect_question_type(self, query: str) -> QuestionType:
        """Detect question type for answer optimization."""
        query_lower = query.lower()
        
        # Check patterns in priority order
        for question_type, patterns in self._question_patterns.items():
            if question_type == QuestionType.OPEN_ENDED:
                continue  # Save as fallback
                
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return question_type
        
        # Special case for yes/no questions
        if any(query_lower.startswith(word) for word in ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should']):
            return QuestionType.YES_NO
        
        return QuestionType.OPEN_ENDED

    async def _extract_entities(self, query: str) -> List[QueryEntity]:
        """Extract entities using configured processing mode."""
        entities = []
        
        # Pattern-based extraction (always performed)
        pattern_entities = self._extract_entities_patterns(query)
        entities.extend(pattern_entities)
        
        # spaCy extraction for BALANCED and COMPREHENSIVE modes
        if self.config.processing_mode != ProcessingMode.FAST and self.spacy_ready:
            spacy_entities = await self._extract_entities_spacy(query)
            entities.extend(spacy_entities)
        
        # Merge and deduplicate entities
        merged_entities = self._merge_entities(entities)
        
        # Filter by confidence threshold
        filtered_entities = [
            entity for entity in merged_entities
            if entity.confidence >= self.config.entity_confidence_threshold
        ]
        
        return filtered_entities

    def _extract_entities_patterns(self, query: str) -> List[QueryEntity]:
        """Extract entities using pattern matching."""
        entities = []
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, query, re.IGNORECASE):
                    entity = QueryEntity(
                        text=match.group(),
                        entity_type=entity_type,
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence=0.8,  # Pattern-based confidence
                        context=query[max(0, match.start()-15):match.end()+15],
                        variants=[],
                        domain_relevance=0.7
                    )
                    entities.append(entity)
        
        return entities

    async def _extract_entities_spacy(self, query: str) -> List[QueryEntity]:
        """Extract entities using spaCy NER."""
        if not self.spacy_ready:
            return []
        
        try:
            doc = self.nlp(query)
            entities = []
            
            for ent in doc.ents:
                # Map spaCy labels to our entity types
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type:
                    entity = QueryEntity(
                        text=ent.text,
                        entity_type=entity_type,
                        start_position=ent.start_char,
                        end_position=ent.end_char,
                        confidence=0.9,  # spaCy has generally high confidence
                        context=query[max(0, ent.start_char-15):ent.end_char+15],
                        variants=[],
                        domain_relevance=0.8,
                        metadata={'spacy_label': ent.label_}
                    )
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            self.logger.warning(f"⚠️ spaCy entity extraction failed: {e}")
            return []

    def _map_spacy_label(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our entity types."""
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'LOC': EntityType.LOCATION,
            'DATE': EntityType.DATE_TIME,
            'TIME': EntityType.DATE_TIME,
            'MONEY': EntityType.CURRENCY,
            'CARDINAL': EntityType.NUMERIC,
            'PERCENT': EntityType.NUMERIC,
            'ORDINAL': EntityType.NUMERIC
        }
        return mapping.get(spacy_label)

    def _merge_entities(self, entities: List[QueryEntity]) -> List[QueryEntity]:
        """Merge and deduplicate extracted entities."""
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda e: e.start_position)
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # Check for overlap or duplication
            if (self._entities_overlap(current, next_entity) or 
                self._entities_duplicate(current, next_entity)):
                # Merge entities, keeping the one with higher confidence
                if next_entity.confidence > current.confidence:
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged

    def _entities_overlap(self, entity1: QueryEntity, entity2: QueryEntity) -> bool:
        """Check if two entities have overlapping text positions."""
        return not (entity1.end_position <= entity2.start_position or 
                   entity2.end_position <= entity1.start_position)

    def _entities_duplicate(self, entity1: QueryEntity, entity2: QueryEntity) -> bool:
        """Check if two entities represent the same information."""
        return (entity1.text.lower() == entity2.text.lower() and 
                entity1.entity_type == entity2.entity_type)

    async def _expand_query(
        self, 
        query: str, 
        entities: List[QueryEntity],
        document_context: Optional[List[DocumentStructure]] = None
    ) -> QueryExpansion:
        """Generate conservative query expansion for precision."""
        if not self.config.enable_query_expansion:
            return QueryExpansion(original_terms=query.split())
        
        # Extract key terms from query
        original_terms = self._extract_key_terms(query)
        
        expansion = QueryExpansion(
            original_terms=original_terms,
            expansion_confidence=0.0
        )
        
        # Generate synonyms (conservative approach)
        if self.nltk_ready:
            expansion.synonyms = await self._generate_synonyms_conservative(original_terms)
        
        # Extract domain terms from document context
        if document_context and self.document_analyzer:
            expansion.domain_terms = self._extract_domain_terms(query, document_context)
        
        # Generate temporal context for date entities
        date_entities = [e for e in entities if e.entity_type == EntityType.DATE_TIME]
        if date_entities:
            expansion.temporal_context = self._generate_temporal_context(date_entities)
        
        # Calculate expansion confidence
        expansion.expansion_confidence = self._calculate_expansion_confidence(expansion)
        
        # Generate suggested filters based on entities
        expansion.suggested_filters = self._generate_suggested_filters(entities)
        
        return expansion

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for expansion."""
        if self.nltk_ready:
            try:
                tokens = word_tokenize(query.lower())
                # Remove stop words and punctuation
                key_terms = [
                    self.lemmatizer.lemmatize(token) 
                    for token in tokens 
                    if (token.isalpha() and 
                        token not in self.stop_words and 
                        len(token) > 2)
                ]
                return key_terms[:10]  # Limit for conservative expansion
            except:
                pass
        
        # Fallback: simple word splitting
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        return [w for w in words if len(w) > 2][:10]

    async def _generate_synonyms_conservative(self, terms: List[str]) -> Dict[str, List[str]]:
        """Generate conservative synonym expansion using WordNet."""
        if not self.nltk_ready:
            return {}
        
        synonyms = {}
        max_synonyms = self.config.expansion_aggressiveness * 10  # 0.3 * 10 = 3 max
        
        try:
            from nltk.corpus import wordnet
            
            for term in terms[:5]:  # Limit terms to process
                cache_key = f"synonyms_{term}"
                
                # Check expansion cache
                if (self.config.cache_config and 
                    self.config.cache_config.cache_expansion_lookups and
                    cache_key in self._expansion_cache):
                    synonyms[term] = self._expansion_cache[cache_key]
                    continue
                
                term_synonyms = []
                synsets = wordnet.synsets(term)
                
                for synset in synsets[:2]:  # Limit synsets for precision
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if (synonym.lower() != term.lower() and 
                            synonym not in term_synonyms and
                            len(synonym) > 2):
                            term_synonyms.append(synonym)
                
                # Conservative filtering: only most relevant synonyms
                if term_synonyms:
                    synonyms[term] = term_synonyms[:int(max_synonyms)]
                    
                    # Cache the result
                    if (self.config.cache_config and 
                        self.config.cache_config.cache_expansion_lookups):
                        self._expansion_cache[cache_key] = synonyms[term]
            
            return synonyms
            
        except Exception as e:
            self.logger.warning(f"⚠️ Synonym generation failed: {e}")
            return {}

    def _extract_domain_terms(
        self, 
        query: str, 
        document_context: List[DocumentStructure]
    ) -> List[str]:
        """Extract domain-specific terms from document context."""
        domain_terms = []
        query_lower = query.lower()
        
        # Analyze entities across document context
        all_entities = []
        for doc_structure in document_context[:10]:  # Limit for performance
            all_entities.extend(doc_structure.entities)
        
        # Find entities relevant to the query
        for entity in all_entities:
            if any(word in entity.text.lower() for word in query_lower.split()):
                if entity.text not in domain_terms:
                    domain_terms.append(entity.text)
        
        # Limit domain terms for conservative expansion
        return domain_terms[:self.expansion.max_domain_terms]

    def _generate_temporal_context(self, date_entities: List[QueryEntity]) -> List[str]:
        """Generate temporal context for date-related queries."""
        temporal_context = []
        
        for entity in date_entities:
            text = entity.text.lower()
            
            # Generate related time periods
            if 'q1' in text or 'first quarter' in text:
                temporal_context.extend(['january', 'february', 'march'])
            elif 'q2' in text or 'second quarter' in text:
                temporal_context.extend(['april', 'may', 'june'])
            elif 'q3' in text or 'third quarter' in text:
                temporal_context.extend(['july', 'august', 'september'])
            elif 'q4' in text or 'fourth quarter' in text:
                temporal_context.extend(['october', 'november', 'december'])
            
            # Extract year for related periods
            year_match = re.search(r'\b(20\d{2})\b', text)
            if year_match:
                year = int(year_match.group(1))
                temporal_context.extend([str(year-1), str(year+1)])  # Adjacent years
        
        return list(set(temporal_context))  # Remove duplicates

    def _calculate_expansion_confidence(self, expansion: QueryExpansion) -> float:
        """Calculate confidence score for query expansion."""
        score = 0.0
        total_weight = 0.0
        
        # Synonym confidence
        if expansion.synonyms:
            synonym_score = min(len(expansion.synonyms) / 5.0, 1.0)  # Max 5 terms
            score += synonym_score * 0.4
            total_weight += 0.4
        
        # Domain terms confidence
        if expansion.domain_terms:
            domain_score = min(len(expansion.domain_terms) / 7.0, 1.0)  # Max 7 terms
            score += domain_score * 0.3
            total_weight += 0.3
        
        # Related concepts confidence
        if expansion.related_concepts:
            concept_score = min(len(expansion.related_concepts) / 5.0, 1.0)
            score += concept_score * 0.2
            total_weight += 0.2
        
        # Temporal context confidence
        if expansion.temporal_context:
            temporal_score = min(len(expansion.temporal_context) / 4.0, 1.0)
            score += temporal_score * 0.1
            total_weight += 0.1
        
        return score / total_weight if total_weight > 0 else 0.0

    def _generate_suggested_filters(self, entities: List[QueryEntity]) -> Dict[str, Any]:
        """Generate suggested filters for retrieval based on entities."""
        filters = {}
        
        # Date range filters
        date_entities = [e for e in entities if e.entity_type == EntityType.DATE_TIME]
        if date_entities:
            filters['date_range'] = [e.text for e in date_entities]
        
        # Document type filters
        doc_type_entities = [e for e in entities if e.entity_type == EntityType.DOCUMENT_TYPE]
        if doc_type_entities:
            filters['document_types'] = [e.text for e in doc_type_entities]
        
        # Organization filters
        org_entities = [e for e in entities if e.entity_type == EntityType.ORGANIZATION]
        if org_entities:
            filters['organizations'] = [e.text for e in org_entities]
        
        return filters

    def _calculate_complexity(self, query: str, entities: List[QueryEntity]) -> float:
        """Calculate query complexity score."""
        complexity = 0.0
        
        # Base complexity from query length
        word_count = len(query.split())
        complexity += min(word_count / 20.0, 0.5)  # Max 0.5 from length
        
        # Entity complexity
        complexity += min(len(entities) / 10.0, 0.3)  # Max 0.3 from entities
        
        # Question complexity indicators
        complex_indicators = ['compare', 'analyze', 'why', 'how', 'relationship', 'impact']
        for indicator in complex_indicators:
            if indicator in query.lower():
                complexity += 0.05
        
        return min(complexity, 1.0)

    def _calculate_confidence(
        self, 
        intent: QueryIntent, 
        entities: List[QueryEntity], 
        expansion: QueryExpansion
    ) -> float:
        """Calculate overall analysis confidence."""
        confidence = 0.0
        
        # Intent confidence (based on processing mode and patterns matched)
        if self.config.processing_mode == ProcessingMode.COMPREHENSIVE:
            confidence += 0.4
        elif self.config.processing_mode == ProcessingMode.BALANCED:
            confidence += 0.35
        else:
            confidence += 0.25
        
        # Entity confidence (average of individual entity confidences)
        if entities:
            avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
            confidence += avg_entity_confidence * 0.3
        
        # Expansion confidence
        confidence += expansion.expansion_confidence * 0.3
        
        return min(confidence, 1.0)

    def _suggest_retrieval_strategies(
        self,
        intent: QueryIntent,
        question_type: QuestionType,
        entities: List[QueryEntity]
    ) -> List[RetrievalStrategy]:
        """Suggest optimal retrieval strategies based on analysis."""
        strategies = []
        
        # Intent-based strategy selection
        if intent == QueryIntent.FACTUAL:
            strategies.append(RetrievalStrategy.ENTITY_FOCUSED)
            if any(e.entity_type == EntityType.NUMERIC for e in entities):
                strategies.append(RetrievalStrategy.TABLE_PRIORITY)
        
        elif intent == QueryIntent.COMPARATIVE:
            strategies.extend([
                RetrievalStrategy.MULTI_DOCUMENT,
                RetrievalStrategy.STRUCTURE_AWARE
            ])
        
        elif intent == QueryIntent.ANALYTICAL:
            strategies.extend([
                RetrievalStrategy.SEMANTIC_SEARCH,
                RetrievalStrategy.STRUCTURE_AWARE
            ])
        
        elif intent == QueryIntent.PROCEDURAL:
            strategies.append(RetrievalStrategy.STRUCTURE_AWARE)
        
        # Entity-based strategy adjustments
        if any(e.entity_type == EntityType.DATE_TIME for e in entities):
            strategies.append(RetrievalStrategy.TEMPORAL_FILTERING)
        
        # Default fallback
        if not strategies:
            strategies.append(RetrievalStrategy.SEMANTIC_SEARCH)
        
        return strategies

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        return {
            'analysis_count': self._analysis_count,
            'average_processing_time_ms': (
                self._total_processing_time / self._analysis_count 
                if self._analysis_count > 0 else 0
            ),
            'cache_size': len(self._cache),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'nlp_status': {
                'nltk_ready': self.nltk_ready,
                'spacy_ready': self.spacy_ready
            },
            'processing_mode': self.config.processing_mode.value
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate for performance monitoring."""
        # This would be implemented with proper hit/miss tracking
        # For now, return estimated value
        return 0.15 if self.config.cache_config and self.config.cache_config.enabled else 0.0

    def clear_cache(self):
        """Clear all cached results."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._expansion_cache.clear()
        self.logger.info("🧹 QueryAnalyzer cache cleared")

    async def batch_analyze(
        self, 
        queries: List[str],
        context: Optional[Dict] = None
    ) -> List[QueryAnalysis]:
        """Analyze multiple queries efficiently."""
        tasks = [
            self.analyze_query(query, context) 
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Batch analysis failed for query {i}: {result}")
                # Create fallback analysis
                fallback = QueryAnalysis(
                    query_id="",
                    original_query=queries[i],
                    processed_query=queries[i],
                    intent=QueryIntent.EXPLORATORY,
                    question_type=QuestionType.OPEN_ENDED,
                    entities=[],
                    expansion=QueryExpansion(original_terms=queries[i].split()),
                    complexity_score=0.5,
                    confidence_score=0.1,
                    suggested_strategies=[RetrievalStrategy.SEMANTIC_SEARCH],
                    processing_time=0.0,
                    processing_mode=ProcessingMode.FAST,
                    metadata={'batch_error': str(result)}
                )
                analyses.append(fallback)
            else:
                analyses.append(result)
        
        return analyses
