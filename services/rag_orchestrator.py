"""
RAG Orchestrator - Complete Pipeline Integration

This service orchestrates the complete RAG pipeline from query to final response,
integrating all phases: PDF processing, query analysis, intelligent retrieval, 
and intelligent summarization with LLaMA.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

# Import all major services
from processing.pdf_processor import PDFProcessor
from processing.ocr_processor import OCRProcessor
from services.text_processor import TextProcessor
from services.document_analyzer import DocumentAnalyzer
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.query_analyzer import QueryAnalyzer
from services.intelligent_retriever import IntelligentRetriever
from services.intelligent_summarizer import (
    IntelligentSummarizer, 
    SummarizationConfig, 
    StreamChunk,
    create_streaming_summarization_config
)
from storage.vector_store import VectorStore

# Import models
from models.query_models import QueryAnalysis
from models.retrieval_models import RetrievalResult


class PipelineMode(Enum):
    """Complete pipeline processing modes"""
    SPEED = "speed"           # <1s total processing
    BALANCED = "balanced"     # <3s total processing  
    COMPREHENSIVE = "comprehensive"  # <10s total processing


class ResponseFormat(Enum):
    """Response format options"""
    TEXT_ONLY = "text_only"
    WITH_SOURCES = "with_sources"
    WITH_ANALYSIS = "with_analysis"
    FULL_DEBUG = "full_debug"


@dataclass
class RAGConfig:
    """Complete RAG pipeline configuration"""
    # Pipeline mode
    mode: PipelineMode = PipelineMode.BALANCED
    
    # Response configuration
    response_format: ResponseFormat = ResponseFormat.WITH_SOURCES
    max_response_length: int = 512
    include_reasoning: bool = True
    
    # Retrieval configuration  
    max_retrieved_chunks: int = 10
    relevance_threshold: float = 0.3
    
    # Summarization configuration
    summarization_config: Optional[SummarizationConfig] = None
    summarization_mode: Any = None  # SummarizationMode from intelligent_summarizer
    
    # Streaming configuration
    enable_streaming: bool = False
    stream_chunk_size: int = 50
    stream_delay_ms: int = 50
    
    # Performance thresholds
    max_processing_time_seconds: float = 10.0
    enable_caching: bool = True


@dataclass
class RAGResponse:
    """Complete RAG response with all metadata"""
    # Main response
    answer: str
    confidence_score: float
    
    # Source information
    sources_used: List[str]
    citations: List[Dict[str, Any]]
    
    # Processing metadata
    total_processing_time_ms: float
    query_analysis_time_ms: float
    retrieval_time_ms: float
    summarization_time_ms: float
    
    # Quality metrics
    retrieval_precision: float
    response_coherence: float
    factual_consistency: float
    
    # Debug information (optional)
    query_analysis: Optional[QueryAnalysis] = None
    retrieval_result: Optional[RetrievalResult] = None
    reasoning_steps: List[str] = None
    
    # Performance breakdown
    chunks_processed: int = 0
    tokens_generated: int = 0
    cache_hits: int = 0
    
    # Streaming metadata
    is_streaming: bool = False
    stream_id: Optional[str] = None


class RAGOrchestrator:
    """
    Complete RAG Pipeline Orchestrator
    
    Orchestrates the end-to-end RAG pipeline:
    1. Query Analysis (Phase 3.3c)
    2. Intelligent Retrieval (Phase 3.3d) 
    3. Intelligent Summarization (Phase 3.4)
    
    Provides unified interface for document processing and question answering.
    """
    
    def __init__(self,
                 vector_store: VectorStore,
                 llama_model_path: Optional[str] = None):
        """Initialize the RAG orchestrator with all required services"""
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize all services
        self.pdf_processor = PDFProcessor()
        self.ocr_processor = OCRProcessor()
        self.text_processor = TextProcessor()
        self.document_analyzer = DocumentAnalyzer()
        self.chunking_service = ChunkingService()
        self.embedding_service = EmbeddingService()
        self.vector_store = vector_store
        
        # Initialize AI services
        self.query_analyzer = QueryAnalyzer()
        self.intelligent_retriever = IntelligentRetriever(
            vector_store=vector_store,
            embedding_service=self.embedding_service,
            document_analyzer=self.document_analyzer
        )
        self.intelligent_summarizer = IntelligentSummarizer(
            model_path=llama_model_path
        )
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        self.success_count = 0
        self.cache = {}  # Simple response cache
        
        self.logger.info("RAG Orchestrator initialized successfully")
    
    async def process_documents(self, 
                              document_paths: List[Union[str, Path]],
                              collection_name: str = "default") -> Dict[str, Any]:
        """
        Process documents through the complete pipeline
        
        Args:
            document_paths: List of document file paths
            collection_name: Vector store collection name
            
        Returns:
            Processing results with statistics
        """
        start_time = time.time()
        processing_stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'errors': [],
            'processing_time_ms': 0
        }
        
        try:
            self.logger.info(f"Starting document processing for {len(document_paths)} documents")
            
            all_chunks = []
            
            for doc_path in document_paths:
                try:
                    doc_path = Path(doc_path)
                    self.logger.info(f"Processing document: {doc_path.name}")
                    
                    # Phase 1: Extract text from PDF
                    if doc_path.suffix.lower() == '.pdf':
                        extraction_result = await self.pdf_processor.extract_text_async(str(doc_path))
                        raw_text = extraction_result.text
                        doc_metadata = extraction_result.metadata
                    else:
                        # Handle other document types
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            raw_text = f.read()
                        doc_metadata = {'source': str(doc_path)}
                    
                    # Phase 2: Process and enhance text
                    processed_result = await self.text_processor.process_text(
                        raw_text, 
                        source_info={'filename': doc_path.name}
                    )
                    
                    # Phase 3: Analyze document structure  
                    analysis_result = await self.document_analyzer.analyze_document(
                        processed_result.processed_text,
                        metadata=doc_metadata
                    )
                    
                    # Phase 4: Create intelligent chunks
                    chunking_result = await self.chunking_service.create_enhanced_chunks(
                        processed_result.processed_text,
                        analysis_result,
                        chunk_size=512,
                        overlap=50
                    )
                    
                    # Add source metadata to chunks
                    for chunk in chunking_result.chunks:
                        chunk.metadata.update({
                            'source': str(doc_path),
                            'document_type': analysis_result.document_type,
                            'quality_score': processed_result.quality_metrics.overall_score
                        })
                    
                    all_chunks.extend(chunking_result.chunks)
                    processing_stats['documents_processed'] += 1
                    processing_stats['chunks_created'] += len(chunking_result.chunks)
                    
                    self.logger.info(f"Processed {doc_path.name}: {len(chunking_result.chunks)} chunks created")
                    
                except Exception as e:
                    error_msg = f"Error processing {doc_path}: {str(e)}"
                    self.logger.error(error_msg)
                    processing_stats['errors'].append(error_msg)
            
            # Phase 5: Generate embeddings and store
            if all_chunks:
                self.logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
                
                # Generate embeddings in batches
                batch_size = 50
                for i in range(0, len(all_chunks), batch_size):
                    batch_chunks = all_chunks[i:i+batch_size]
                    
                    embeddings = await self.embedding_service.generate_embeddings_batch([
                        chunk.content for chunk in batch_chunks
                    ])
                    
                    # Store chunks with embeddings
                    await self.vector_store.add_chunks_with_embeddings(
                        batch_chunks,
                        embeddings,
                        collection_name=collection_name
                    )
                    
                    processing_stats['embeddings_generated'] += len(embeddings)
            
            processing_stats['processing_time_ms'] = (time.time() - start_time) * 1000
            
            self.logger.info(f"Document processing completed: {processing_stats}")
            return processing_stats
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            processing_stats['processing_time_ms'] = (time.time() - start_time) * 1000
            processing_stats['errors'].append(f"Pipeline error: {str(e)}")
            return processing_stats
    
    async def ask_question(self,
                          query: str,
                          collection_name: str = "default", 
                          config: Optional[RAGConfig] = None) -> RAGResponse:
        """
        Complete question answering pipeline
        
        Args:
            query: User question
            collection_name: Vector store collection to search
            config: RAG configuration
            
        Returns:
            Complete RAG response with answer and metadata
        """
        start_time = time.time()
        config = config or RAGConfig()
        
        # Check cache first
        cache_key = f"{query}_{collection_name}_{config.mode.value}"
        if config.enable_caching and cache_key in self.cache:
            cached_response = self.cache[cache_key]
            cached_response.cache_hits = 1
            return cached_response
        
        try:
            self.logger.info(f"Processing query: {query[:100]}...")
            
            timing = {}
            reasoning_steps = []
            
            # Phase 1: Query Analysis
            analysis_start = time.time()
            
            # Configure query analyzer based on pipeline mode
            if config.mode == PipelineMode.SPEED:
                analysis_mode = "FAST"
            elif config.mode == PipelineMode.COMPREHENSIVE:
                analysis_mode = "COMPREHENSIVE"  
            else:
                analysis_mode = "BALANCED"
            
            query_analysis = await self.query_analyzer.analyze_query(
                query, 
                processing_mode=analysis_mode
            )
            
            timing['query_analysis_ms'] = (time.time() - analysis_start) * 1000
            reasoning_steps.append(f"Analyzed query intent: {query_analysis.intent.name}")
            reasoning_steps.append(f"Extracted {len(query_analysis.entities)} entities")
            
            # Phase 2: Intelligent Retrieval
            retrieval_start = time.time()
            
            # Configure retrieval based on pipeline mode
            if config.mode == PipelineMode.SPEED:
                retrieval_mode = "SPEED_OPTIMIZED"
            elif config.mode == PipelineMode.COMPREHENSIVE:
                retrieval_mode = "ACCURACY_OPTIMIZED"
            else:
                retrieval_mode = "BALANCED"
            
            retrieval_result = await self.intelligent_retriever.retrieve_intelligent(
                query_analysis,
                collection_name=collection_name,
                top_k=config.max_retrieved_chunks,
                mode=retrieval_mode
            )
            
            timing['retrieval_ms'] = (time.time() - retrieval_start) * 1000
            reasoning_steps.append(f"Retrieved {len(retrieval_result.ranked_chunks)} relevant chunks")
            reasoning_steps.append(f"Average relevance score: {retrieval_result.average_score:.3f}")
            
            # Phase 3: Intelligent Summarization
            summarization_start = time.time()
            
            # Configure summarization
            summ_config = config.summarization_config or self._get_default_summarization_config(config.mode)
            summ_config.max_response_length = config.max_response_length
            
            summary_result = await self.intelligent_summarizer.summarize_with_context(
                query,
                query_analysis, 
                retrieval_result,
                summ_config
            )
            
            timing['summarization_ms'] = (time.time() - summarization_start) * 1000
            reasoning_steps.extend(summary_result.reasoning_steps)
            
            # Create final response
            total_time = (time.time() - start_time) * 1000
            
            response = RAGResponse(
                answer=summary_result.response_text,
                confidence_score=summary_result.confidence_score,
                sources_used=summary_result.sources_used,
                citations=summary_result.citations,
                total_processing_time_ms=total_time,
                query_analysis_time_ms=timing['query_analysis_ms'],
                retrieval_time_ms=timing['retrieval_ms'], 
                summarization_time_ms=timing['summarization_ms'],
                retrieval_precision=retrieval_result.average_score,
                response_coherence=summary_result.coherence_score,
                factual_consistency=summary_result.factual_consistency_score,
                chunks_processed=len(retrieval_result.ranked_chunks),
                tokens_generated=summary_result.tokens_generated,
                cache_hits=0
            )
            
            # Add debug information based on response format
            if config.response_format in [ResponseFormat.WITH_ANALYSIS, ResponseFormat.FULL_DEBUG]:
                response.query_analysis = query_analysis
                response.reasoning_steps = reasoning_steps
                
            if config.response_format == ResponseFormat.FULL_DEBUG:
                response.retrieval_result = retrieval_result
            
            # Cache response
            if config.enable_caching:
                self.cache[cache_key] = response
            
            # Update metrics
            self._update_metrics(total_time, True)
            
            self.logger.info(f"Query processed successfully in {total_time:.1f}ms")
            return response
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            self.logger.error(f"Query processing failed: {e}")
            
            # Create error response
            error_response = RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                confidence_score=0.0,
                sources_used=[],
                citations=[],
                total_processing_time_ms=total_time,
                query_analysis_time_ms=0.0,
                retrieval_time_ms=0.0,
                summarization_time_ms=0.0,
                retrieval_precision=0.0,
                response_coherence=0.0,
                factual_consistency=0.0,
                reasoning_steps=[f"Error: {str(e)}"]
            )
            
            self._update_metrics(total_time, False)
            return error_response
    
    async def ask_question_streaming(self,
                                   query: str,
                                   collection_name: str = "default",
                                   config: Optional[RAGConfig] = None) -> AsyncGenerator[Union[StreamChunk, RAGResponse], None]:
        """
        Streaming version of question answering pipeline
        
        Args:
            query: User question
            collection_name: Vector store collection to search
            config: RAG configuration with streaming enabled
            
        Yields:
            StreamChunk objects with partial responses and final RAGResponse
        """
        start_time = time.time()
        config = config or RAGConfig()
        
        try:
            self.logger.info(f"Processing streaming query: {query[:100]}...")
            
            timing = {}
            reasoning_steps = []
            
            # Send initial status chunk
            yield StreamChunk(
                chunk_id=0,
                content="",
                metadata={
                    "type": "status",
                    "message": "Starting query processing...",
                    "stage": "initialization"
                }
            )
            
            # Phase 1: Query Analysis
            yield StreamChunk(
                chunk_id=1,
                content="",
                metadata={
                    "type": "status",
                    "message": "Analyzing your question...",
                    "stage": "analysis"
                }
            )
            
            analysis_start = time.time()
            
            if config.mode == PipelineMode.SPEED:
                analysis_mode = "FAST"
            elif config.mode == PipelineMode.COMPREHENSIVE:
                analysis_mode = "COMPREHENSIVE"  
            else:
                analysis_mode = "BALANCED"
            
            query_analysis = await self.query_analyzer.analyze_query(
                query, 
                processing_mode=analysis_mode
            )
            
            timing['query_analysis_ms'] = (time.time() - analysis_start) * 1000
            reasoning_steps.append(f"Analyzed query intent: {query_analysis.intent.name}")
            
            # Phase 2: Intelligent Retrieval
            yield StreamChunk(
                chunk_id=2,
                content="",
                metadata={
                    "type": "status",
                    "message": "Searching relevant content...",
                    "stage": "retrieval",
                    "intent": query_analysis.intent.name
                }
            )
            
            retrieval_start = time.time()
            
            # Configure retrieval based on mode
            max_chunks = min(config.max_retrieved_chunks, 15 if config.mode == PipelineMode.SPEED else 25)
            
            retrieval_result = await self.intelligent_retriever.retrieve_with_analysis(
                query_analysis,
                collection_name=collection_name,
                max_results=max_chunks
            )
            
            timing['retrieval_ms'] = (time.time() - retrieval_start) * 1000
            reasoning_steps.append(f"Retrieved {len(retrieval_result.ranked_chunks)} relevant chunks")
            
            # Phase 3: Streaming Summarization
            yield StreamChunk(
                chunk_id=3,
                content="",
                metadata={
                    "type": "status",
                    "message": "Generating response...",
                    "stage": "summarization",
                    "chunks_found": len(retrieval_result.ranked_chunks)
                }
            )
            
            summarization_start = time.time()
            
            # Create streaming summarization config
            summarization_config = create_streaming_summarization_config(
                mode=config.summarization_mode,
                chunk_size=50 if config.mode == PipelineMode.SPEED else 70,
                delay_ms=30 if config.mode == PipelineMode.SPEED else 50
            )
            
            # Configure based on response format
            summarization_config.include_citations = config.response_format in [
                ResponseFormat.WITH_SOURCES, 
                ResponseFormat.WITH_ANALYSIS,
                ResponseFormat.FULL_DEBUG
            ]
            
            # Stream the response
            chunk_id = 4
            response_chunks = []
            
            # Get streaming response from summarizer
            stream_response = await self.intelligent_summarizer.summarize_with_context(
                query, query_analysis, retrieval_result, summarization_config
            )
            
            # Forward streaming chunks
            async for chunk in stream_response:
                # Track content chunks for final response
                if chunk.metadata and chunk.metadata.get("type") == "content":
                    response_chunks.append(chunk.content)
                
                # Forward the chunk with updated ID
                chunk.chunk_id = chunk_id
                yield chunk
                chunk_id += 1
            
            timing['summarization_ms'] = (time.time() - summarization_start) * 1000
            
            # Send final completion status
            total_time = (time.time() - start_time) * 1000
            
            # Assemble final response text
            full_response_text = "".join(response_chunks)
            
            # Create final RAGResponse for compatibility
            final_response = RAGResponse(
                answer=full_response_text,
                confidence_score=0.8,  # Default confidence for streaming
                sources_used=[chunk.source for chunk in retrieval_result.ranked_chunks[:5]],
                citations=[],  # Citations handled in streaming chunks
                total_processing_time_ms=total_time,
                query_analysis_time_ms=timing.get('query_analysis_ms', 0.0),
                retrieval_time_ms=timing.get('retrieval_ms', 0.0),
                summarization_time_ms=timing.get('summarization_ms', 0.0),
                retrieval_precision=retrieval_result.overall_precision,
                response_coherence=0.8,
                factual_consistency=0.8,
                reasoning_steps=reasoning_steps,
                query_analysis=query_analysis.to_dict() if config.response_format in [
                    ResponseFormat.WITH_ANALYSIS, ResponseFormat.FULL_DEBUG
                ] else None,
                chunk_count=len(retrieval_result.ranked_chunks),
                tokens_generated=len(full_response_text.split()),
                is_streaming=True
            )
            
            # Update metrics
            self._update_metrics(total_time, True)
            
            # Yield final response
            yield final_response
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            self.logger.error(f"Streaming query processing failed: {e}")
            
            # Send error chunk
            yield StreamChunk(
                chunk_id=999,
                content=f"Error: {str(e)}",
                is_final=True,
                metadata={
                    "type": "error",
                    "error": str(e)
                }
            )
            
            self._update_metrics(total_time, False)
    
    async def ask_multiple_questions(self,
                                   questions: List[str],
                                   collection_name: str = "default",
                                   config: Optional[RAGConfig] = None) -> List[RAGResponse]:
        """Process multiple questions efficiently"""
        
        # Process questions concurrently for better performance
        tasks = [
            self.ask_question(question, collection_name, config)
            for question in questions
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_response = RAGResponse(
                    answer=f"Error processing question {i+1}: {str(response)}",
                    confidence_score=0.0,
                    sources_used=[],
                    citations=[],
                    total_processing_time_ms=0.0,
                    query_analysis_time_ms=0.0,
                    retrieval_time_ms=0.0,
                    summarization_time_ms=0.0,
                    retrieval_precision=0.0,
                    response_coherence=0.0,
                    factual_consistency=0.0
                )
                final_responses.append(error_response)
            else:
                final_responses.append(response)
        
        return final_responses
    
    def _get_default_summarization_config(self, mode: PipelineMode) -> SummarizationConfig:
        """Get default summarization config based on pipeline mode"""
        
        if mode == PipelineMode.SPEED:
            return SummarizationConfig(
                mode="FAST",
                max_response_length=256,
                temperature=0.3,
                include_citations=False
            )
        elif mode == PipelineMode.COMPREHENSIVE:
            return SummarizationConfig(
                mode="COMPREHENSIVE",
                max_response_length=768,
                temperature=0.8,
                include_citations=True,
                enable_fact_checking=True
            )
        else:  # BALANCED
            return SummarizationConfig(
                mode="BALANCED",
                max_response_length=512,
                temperature=0.7,
                include_citations=True
            )
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update performance metrics"""
        self.request_count += 1
        self.total_processing_time += processing_time
        if success:
            self.success_count += 1
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        base_stats = {
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'success_rate': self.success_count / self.request_count if self.request_count > 0 else 0.0,
            'average_processing_time_ms': self.total_processing_time / self.request_count if self.request_count > 0 else 0.0,
            'cache_size': len(self.cache)
        }
        
        # Get stats from individual services
        query_stats = self.query_analyzer.get_performance_stats()
        retrieval_stats = self.intelligent_retriever.get_performance_stats() 
        summarization_stats = self.intelligent_summarizer.get_performance_stats()
        
        return {
            'orchestrator': base_stats,
            'query_analyzer': query_stats,
            'intelligent_retriever': retrieval_stats,
            'intelligent_summarizer': summarization_stats
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.cache.clear()
        self.query_analyzer.clear_cache() if hasattr(self.query_analyzer, 'clear_cache') else None
        self.intelligent_retriever.clear_cache() if hasattr(self.intelligent_retriever, 'clear_cache') else None
        self.intelligent_summarizer.clear_cache()
        
        self.logger.info("All caches cleared")


# Configuration helpers
def create_speed_rag_config() -> RAGConfig:
    """Create RAG configuration optimized for speed"""
    return RAGConfig(
        mode=PipelineMode.SPEED,
        response_format=ResponseFormat.TEXT_ONLY,
        max_response_length=256,
        max_retrieved_chunks=5,
        relevance_threshold=0.5,
        max_processing_time_seconds=2.0,
        enable_caching=True
    )


def create_balanced_rag_config() -> RAGConfig:
    """Create balanced RAG configuration (default)"""
    return RAGConfig(
        mode=PipelineMode.BALANCED,
        response_format=ResponseFormat.WITH_SOURCES,
        max_response_length=512,
        max_retrieved_chunks=10,
        relevance_threshold=0.3,
        max_processing_time_seconds=5.0,
        enable_caching=True
    )


def create_comprehensive_rag_config() -> RAGConfig:
    """Create comprehensive RAG configuration for maximum quality"""
    return RAGConfig(
        mode=PipelineMode.COMPREHENSIVE,
        response_format=ResponseFormat.WITH_ANALYSIS,
        max_response_length=768,
        max_retrieved_chunks=15,
        relevance_threshold=0.2,
        max_processing_time_seconds=15.0,
        enable_caching=True,
        include_reasoning=True
    )


def create_debug_rag_config() -> RAGConfig:
    """Create debug RAG configuration with full introspection"""
    return RAGConfig(
        mode=PipelineMode.COMPREHENSIVE,
        response_format=ResponseFormat.FULL_DEBUG,
        max_response_length=768,
        max_retrieved_chunks=20,
        relevance_threshold=0.1,
        max_processing_time_seconds=20.0,
        enable_caching=False,  # Disable caching for debugging
        include_reasoning=True
    )
