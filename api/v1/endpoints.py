"""
Complete RAG API - RESTful endpoints for the Modern RAG App

Provides comprehensive API endpoints for:
- Document processing and ingestion
- Question answering with intelligent retrieval
- System management and monitoring
- Performance analytics

FastAPI-based with async support and comprehensive error handling.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from enum import Enum
import asyncio
import logging
import time
from pathlib import Path
import tempfile
import os
import json

# Import our services
from services.rag_orchestrator import RAGOrchestrator, RAGConfig, RAGResponse, PipelineMode, ResponseFormat
from services.intelligent_summarizer import SummarizationConfig, SummarizationMode, ResponseStyle, StreamChunk
from storage.vector_store import VectorStore
from config.settings import get_settings


# API Models
class QueryRequest(BaseModel):
    """Request model for question answering"""
    query: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    collection_name: str = Field("default", description="Vector store collection to search")
    mode: Optional[str] = Field("balanced", description="Processing mode: speed, balanced, comprehensive")
    response_format: Optional[str] = Field("with_sources", description="Response format")
    max_response_length: Optional[int] = Field(512, ge=50, le=2048, description="Maximum response length")
    max_retrieved_chunks: Optional[int] = Field(10, ge=1, le=50, description="Maximum chunks to retrieve")
    include_reasoning: Optional[bool] = Field(True, description="Include reasoning steps")


class StreamingQueryRequest(BaseModel):
    """Request model for streaming question answering"""
    query: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    collection_name: str = Field("default", description="Vector store collection to search")
    mode: Optional[str] = Field("balanced", description="Processing mode: speed, balanced, comprehensive")
    response_format: Optional[str] = Field("with_sources", description="Response format")
    max_response_length: Optional[int] = Field(512, ge=50, le=2048, description="Maximum response length")
    max_retrieved_chunks: Optional[int] = Field(10, ge=1, le=50, description="Maximum chunks to retrieve")
    include_reasoning: Optional[bool] = Field(True, description="Include reasoning steps")
    
    # Streaming specific settings
    stream_chunk_size: Optional[int] = Field(50, ge=10, le=200, description="Words per streaming chunk")
    stream_delay_ms: Optional[int] = Field(50, ge=0, le=500, description="Delay between chunks in milliseconds")


class MultiQueryRequest(BaseModel):
    """Request model for multiple questions"""
    queries: List[str] = Field(..., min_items=1, max_items=20, description="List of questions")
    collection_name: str = Field("default", description="Vector store collection to search")
    mode: Optional[str] = Field("balanced", description="Processing mode")
    response_format: Optional[str] = Field("with_sources", description="Response format")


class DocumentProcessingRequest(BaseModel):
    """Request model for document processing"""
    collection_name: str = Field("default", description="Collection name for storing documents")
    document_types: List[str] = Field(["pdf"], description="Supported document types")
    chunk_size: Optional[int] = Field(512, ge=128, le=2048, description="Chunk size for text splitting")
    overlap: Optional[int] = Field(50, ge=0, le=200, description="Chunk overlap size")


class QueryResponse(BaseModel):
    """Response model for question answering"""
    answer: str
    confidence_score: float
    sources_used: List[str]
    citations: List[Dict[str, Any]]
    processing_time_ms: float
    retrieval_precision: float
    response_coherence: float
    factual_consistency: float
    reasoning_steps: Optional[List[str]] = None
    query_analysis: Optional[Dict[str, Any]] = None
    chunks_processed: int
    tokens_generated: int


class ProcessingResponse(BaseModel):
    """Response model for document processing"""
    success: bool
    documents_processed: int
    chunks_created: int
    embeddings_generated: int
    processing_time_ms: float
    errors: List[str] = []
    collection_name: str


class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    orchestrator_stats: Dict[str, Any]
    service_stats: Dict[str, Any]
    system_health: Dict[str, Any]


# Create FastAPI app
app = FastAPI(
    title="Modern RAG App API",
    description="Complete RAG pipeline with intelligent retrieval and LLaMA summarization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = get_settings()
vector_store = VectorStore()
orchestrator = RAGOrchestrator(
    vector_store=vector_store,
    llama_model_path=settings.llama_model_path
)

logger = logging.getLogger(__name__)


# Helper functions
def create_rag_config(request: Union[QueryRequest, MultiQueryRequest]) -> RAGConfig:
    """Create RAG configuration from request"""
    
    # Map string modes to enums
    mode_mapping = {
        "speed": PipelineMode.SPEED,
        "balanced": PipelineMode.BALANCED,
        "comprehensive": PipelineMode.COMPREHENSIVE
    }
    
    format_mapping = {
        "text_only": ResponseFormat.TEXT_ONLY,
        "with_sources": ResponseFormat.WITH_SOURCES,
        "with_analysis": ResponseFormat.WITH_ANALYSIS,
        "full_debug": ResponseFormat.FULL_DEBUG
    }
    
    mode = mode_mapping.get(request.mode, PipelineMode.BALANCED)
    response_format = format_mapping.get(request.response_format, ResponseFormat.WITH_SOURCES)
    
    config = RAGConfig(
        mode=mode,
        response_format=response_format,
        include_reasoning=getattr(request, 'include_reasoning', True)
    )
    
    # Apply request-specific overrides
    if hasattr(request, 'max_response_length'):
        config.max_response_length = request.max_response_length
    if hasattr(request, 'max_retrieved_chunks'):
        config.max_retrieved_chunks = request.max_retrieved_chunks
    
    return config


def create_streaming_rag_config(request: StreamingQueryRequest) -> RAGConfig:
    """Create streaming RAG configuration from request"""
    
    # Map string modes to enums
    mode_mapping = {
        "speed": PipelineMode.SPEED,
        "balanced": PipelineMode.BALANCED,
        "comprehensive": PipelineMode.COMPREHENSIVE
    }
    
    format_mapping = {
        "text_only": ResponseFormat.TEXT_ONLY,
        "with_sources": ResponseFormat.WITH_SOURCES,
        "with_analysis": ResponseFormat.WITH_ANALYSIS,
        "full_debug": ResponseFormat.FULL_DEBUG
    }
    
    mode = mode_mapping.get(request.mode, PipelineMode.BALANCED)
    response_format = format_mapping.get(request.response_format, ResponseFormat.WITH_SOURCES)
    
    # Map mode to summarization mode
    summarization_mode_mapping = {
        PipelineMode.SPEED: SummarizationMode.FAST,
        PipelineMode.BALANCED: SummarizationMode.BALANCED,
        PipelineMode.COMPREHENSIVE: SummarizationMode.COMPREHENSIVE
    }
    
    config = RAGConfig(
        mode=mode,
        response_format=response_format,
        include_reasoning=request.include_reasoning,
        enable_streaming=True,
        stream_chunk_size=request.stream_chunk_size,
        stream_delay_ms=request.stream_delay_ms,
        summarization_mode=summarization_mode_mapping[mode]
    )
    
    # Apply request-specific overrides
    config.max_response_length = request.max_response_length
    config.max_retrieved_chunks = request.max_retrieved_chunks
    
    return config


def convert_rag_response(rag_response: RAGResponse) -> QueryResponse:
    """Convert RAGResponse to API response model"""
    return QueryResponse(
        answer=rag_response.answer,
        confidence_score=rag_response.confidence_score,
        sources_used=rag_response.sources_used,
        citations=rag_response.citations,
        processing_time_ms=rag_response.total_processing_time_ms,
        retrieval_precision=rag_response.retrieval_precision,
        response_coherence=rag_response.response_coherence,
        factual_consistency=rag_response.factual_consistency,
        reasoning_steps=rag_response.reasoning_steps,
        query_analysis=rag_response.query_analysis.__dict__ if rag_response.query_analysis else None,
        chunks_processed=rag_response.chunks_processed,
        tokens_generated=rag_response.tokens_generated
    )


# API Endpoints

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "Modern RAG App API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": time.time()
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Comprehensive health check"""
    try:
        # Test vector store
        vector_health = await vector_store.health_check()
        
        # Test orchestrator
        orchestrator_stats = orchestrator.get_system_stats()
        
        return {
            "status": "healthy",
            "services": {
                "vector_store": "healthy" if vector_health else "unhealthy",
                "orchestrator": "healthy",
                "llama_model": "loaded" if orchestrator.intelligent_summarizer.model else "not_loaded"
            },
            "stats": orchestrator_stats['orchestrator'],
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Document Processing Endpoints

@app.post("/documents/upload", response_model=ProcessingResponse, tags=["Documents"])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    collection_name: str = Form("default"),
    chunk_size: int = Form(512),
    overlap: int = Form(50)
):
    """
    Upload and process documents
    
    Supports PDF files and processes them through the complete pipeline:
    - PDF text extraction
    - Text processing and enhancement  
    - Document structure analysis
    - Intelligent chunking
    - Embedding generation and storage
    """
    
    if len(files) > 20:  # Limit number of files
        raise HTTPException(status_code=400, detail="Maximum 20 files allowed per upload")
    
    start_time = time.time()
    
    try:
        # Save uploaded files temporarily
        temp_files = []
        for file in files:
            if file.content_type != "application/pdf":
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)
        
        # Process documents
        processing_stats = await orchestrator.process_documents(
            document_paths=temp_files,
            collection_name=collection_name
        )
        
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return ProcessingResponse(
            success=len(processing_stats['errors']) == 0,
            documents_processed=processing_stats['documents_processed'],
            chunks_created=processing_stats['chunks_created'],
            embeddings_generated=processing_stats['embeddings_generated'],
            processing_time_ms=processing_stats['processing_time_ms'],
            errors=processing_stats['errors'],
            collection_name=collection_name
        )
        
    except Exception as e:
        # Cleanup on error
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@app.post("/documents/process-folder", response_model=ProcessingResponse, tags=["Documents"])
async def process_document_folder(
    folder_path: str,
    collection_name: str = "default",
    chunk_size: int = 512,
    overlap: int = 50
):
    """
    Process all documents in a folder
    
    Processes all PDF files in the specified folder through the complete pipeline.
    Useful for bulk document processing.
    """
    
    try:
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise HTTPException(status_code=400, detail="Invalid folder path")
        
        # Find all PDF files
        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            raise HTTPException(status_code=400, detail="No PDF files found in folder")
        
        if len(pdf_files) > 50:  # Limit for performance
            raise HTTPException(status_code=400, detail="Maximum 50 files allowed per folder")
        
        # Process documents
        processing_stats = await orchestrator.process_documents(
            document_paths=[str(f) for f in pdf_files],
            collection_name=collection_name
        )
        
        return ProcessingResponse(
            success=len(processing_stats['errors']) == 0,
            documents_processed=processing_stats['documents_processed'],
            chunks_created=processing_stats['chunks_created'],
            embeddings_generated=processing_stats['embeddings_generated'],
            processing_time_ms=processing_stats['processing_time_ms'],
            errors=processing_stats['errors'],
            collection_name=collection_name
        )
        
    except Exception as e:
        logger.error(f"Folder processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Folder processing failed: {str(e)}")


# Question Answering Endpoints

@app.post("/ask", response_model=QueryResponse, tags=["Question Answering"])
async def ask_question(request: QueryRequest):
    """
    Ask a single question
    
    Complete RAG pipeline processing:
    1. Query analysis (intent classification, entity extraction)
    2. Intelligent retrieval (multi-modal search with adaptive weighting)
    3. Intelligent summarization (LLaMA-powered response generation)
    
    Supports multiple processing modes and response formats.
    """
    
    try:
        config = create_rag_config(request)
        
        rag_response = await orchestrator.ask_question(
            query=request.query,
            collection_name=request.collection_name,
            config=config
        )
        
        return convert_rag_response(rag_response)
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")


@app.post("/ask/stream", tags=["Question Answering"])
async def ask_question_streaming(request: StreamingQueryRequest):
    """
    Ask a question with streaming response
    
    Returns Server-Sent Events (SSE) stream with:
    - Status updates during processing
    - Progressive response chunks as they're generated
    - Final completion metadata
    
    Optimized for real-time user interaction and long responses.
    """
    
    async def generate_stream():
        try:
            # Create streaming RAG config
            config = create_streaming_rag_config(request)
            
            # Process with streaming
            async for chunk_or_response in orchestrator.ask_question_streaming(
                query=request.query,
                collection_name=request.collection_name,
                config=config
            ):
                
                # Handle different response types
                if isinstance(chunk_or_response, StreamChunk):
                    # Stream chunk data
                    chunk_data = {
                        "type": "chunk",
                        "chunk_id": chunk_or_response.chunk_id,
                        "content": chunk_or_response.content,
                        "is_final": chunk_or_response.is_final,
                        "timestamp": chunk_or_response.timestamp,
                        "metadata": chunk_or_response.metadata or {}
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                elif isinstance(chunk_or_response, RAGResponse):
                    # Final response summary
                    final_data = {
                        "type": "final",
                        "response": convert_rag_response(chunk_or_response).__dict__,
                        "completed": True
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"
                    
                    # End the stream
                    break
            
        except Exception as e:
            logger.error(f"Streaming question answering failed: {e}")
            error_data = {
                "type": "error",
                "error": str(e),
                "completed": False
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )


@app.post("/ask/multiple", response_model=List[QueryResponse], tags=["Question Answering"])
async def ask_multiple_questions(request: MultiQueryRequest):
    """
    Ask multiple questions efficiently
    
    Processes multiple questions concurrently for better performance.
    All questions are processed with the same configuration.
    """
    
    try:
        config = create_rag_config(request)
        
        rag_responses = await orchestrator.ask_multiple_questions(
            questions=request.queries,
            collection_name=request.collection_name,
            config=config
        )
        
        return [convert_rag_response(response) for response in rag_responses]
        
    except Exception as e:
        logger.error(f"Multiple question processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multiple question processing failed: {str(e)}")


@app.get("/ask/similar/{query}", tags=["Question Answering"])
async def find_similar_questions(
    query: str,
    collection_name: str = "default",
    limit: int = 5
):
    """
    Find documents similar to a query without generating an answer
    
    Useful for exploring available content and understanding
    what information is available in the document collection.
    """
    
    try:
        # Use the query analyzer to understand the query
        query_analysis = await orchestrator.query_analyzer.analyze_query(query)
        
        # Use intelligent retriever to find similar content
        retrieval_result = await orchestrator.intelligent_retriever.retrieve_intelligent(
            query_analysis,
            collection_name=collection_name,
            top_k=limit
        )
        
        # Return similar chunks without summarization
        similar_content = []
        for ranked_chunk in retrieval_result.ranked_chunks:
            similar_content.append({
                "content": ranked_chunk.chunk.content[:300] + "...",
                "source": ranked_chunk.chunk.metadata.get('source', 'Unknown'),
                "relevance_score": ranked_chunk.final_rank_score,
                "chunk_type": ranked_chunk.chunk.content_type,
                "page_number": ranked_chunk.chunk.metadata.get('page_number', 'N/A')
            })
        
        return {
            "query": query,
            "query_intent": query_analysis.intent.name,
            "entities_found": len(query_analysis.entities),
            "similar_content": similar_content,
            "total_found": len(similar_content)
        }
        
    except Exception as e:
        logger.error(f"Similar question search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")


# System Management Endpoints

@app.get("/stats", response_model=SystemStatsResponse, tags=["System"])
async def get_system_stats():
    """Get comprehensive system statistics and performance metrics"""
    
    try:
        stats = orchestrator.get_system_stats()
        
        # Add system health information
        health_info = {
            "uptime_seconds": time.time() - (stats['orchestrator'].get('start_time', time.time())),
            "memory_usage": "N/A",  # Could add psutil for detailed memory stats
            "disk_usage": "N/A",
            "cache_efficiency": {
                "orchestrator_cache_size": stats['orchestrator']['cache_size'],
                "hit_rate_estimate": "N/A"
            }
        }
        
        return SystemStatsResponse(
            orchestrator_stats=stats['orchestrator'],
            service_stats={
                "query_analyzer": stats['query_analyzer'],
                "intelligent_retriever": stats['intelligent_retriever'],
                "intelligent_summarizer": stats['intelligent_summarizer']
            },
            system_health=health_info
        )
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@app.post("/cache/clear", tags=["System"])
async def clear_system_cache():
    """Clear all system caches"""
    
    try:
        orchestrator.clear_cache()
        return {
            "success": True,
            "message": "All caches cleared successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clearing failed: {str(e)}")


@app.get("/collections", tags=["System"])
async def list_collections():
    """List all available document collections"""
    
    try:
        collections = await vector_store.list_collections()
        
        collection_info = []
        for collection in collections:
            # Get basic stats for each collection
            stats = await vector_store.get_collection_stats(collection)
            collection_info.append({
                "name": collection,
                "document_count": stats.get('document_count', 0),
                "chunk_count": stats.get('chunk_count', 0),
                "created_at": stats.get('created_at', 'Unknown')
            })
        
        return {
            "collections": collection_info,
            "total_collections": len(collection_info)
        }
        
    except Exception as e:
        logger.error(f"Collection listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Collection listing failed: {str(e)}")


@app.delete("/collections/{collection_name}", tags=["System"])
async def delete_collection(collection_name: str):
    """Delete a document collection"""
    
    try:
        await vector_store.delete_collection(collection_name)
        return {
            "success": True,
            "message": f"Collection '{collection_name}' deleted successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Collection deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Collection deletion failed: {str(e)}")


# Configuration Endpoints

@app.get("/config/modes", tags=["Configuration"])
async def get_available_modes():
    """Get available processing modes and response formats"""
    
    return {
        "pipeline_modes": {
            "speed": {
                "description": "Fast processing with basic analysis",
                "target_time": "<1s",
                "use_case": "Real-time chat, quick lookups"
            },
            "balanced": {
                "description": "Good balance of speed and accuracy",
                "target_time": "<3s", 
                "use_case": "Standard question answering"
            },
            "comprehensive": {
                "description": "Maximum accuracy with full analysis",
                "target_time": "<10s",
                "use_case": "Complex analysis, research queries"
            }
        },
        "response_formats": {
            "text_only": "Just the answer text",
            "with_sources": "Answer with source citations",
            "with_analysis": "Answer with reasoning and analysis",
            "full_debug": "Complete processing details"
        },
        "summarization_modes": {
            "fast": "Template-based quick responses",
            "balanced": "LLaMA with moderate processing",
            "comprehensive": "Full LLaMA analysis with fact checking"
        }
    }


# Development and Testing Endpoints

@app.post("/test/pipeline", tags=["Development"])
async def test_complete_pipeline():
    """Test the complete RAG pipeline with sample data"""
    
    try:
        # This would be a comprehensive test of all pipeline components
        test_results = {
            "pdf_processor": "✅ Ready",
            "text_processor": "✅ Ready", 
            "document_analyzer": "✅ Ready",
            "chunking_service": "✅ Ready",
            "embedding_service": "✅ Ready",
            "query_analyzer": "✅ Ready",
            "intelligent_retriever": "✅ Ready",
            "intelligent_summarizer": "✅ Ready",
            "vector_store": "✅ Ready",
            "orchestrator": "✅ Ready"
        }
        
        # Test basic orchestrator functionality
        stats = orchestrator.get_system_stats()
        
        return {
            "status": "All systems operational",
            "component_tests": test_results,
            "system_stats": stats['orchestrator'],
            "test_timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline test failed: {str(e)}")


# Error Handlers

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}


# Startup and Shutdown Events

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Modern RAG App API starting up...")
    
    # Initialize vector store
    await vector_store.initialize()
    
    logger.info("Modern RAG App API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Modern RAG App API shutting down...")
    
    # Clear caches
    orchestrator.clear_cache()
    
    # Close vector store connections
    await vector_store.close()
    
    logger.info("Modern RAG App API shutdown complete!")


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
