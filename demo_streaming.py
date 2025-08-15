"""
Streaming Demo - Test the new streaming capabilities

This demo showcases the optional streaming feature in the Modern RAG App,
demonstrating real-time response generation with progressive delivery.
"""

import asyncio
import json
from typing import AsyncGenerator
import logging

# Import the streaming components
from services.intelligent_summarizer import (
    IntelligentSummarizer, 
    SummarizationConfig,
    SummarizationMode,
    create_streaming_summarization_config,
    StreamChunk
)
from services.rag_orchestrator import RAGOrchestrator, RAGConfig, PipelineMode
from storage.chroma_store import ChromaVectorStore
from models.query_models import QueryAnalysis, QueryIntent, QueryEntity, QueryComplexity


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingDemo:
    """Demonstration of streaming capabilities"""
    
    def __init__(self):
        """Initialize demo components"""
        self.vector_store = ChromaVectorStore("streaming_demo")
        self.orchestrator = RAGOrchestrator(self.vector_store)
        
        # Mock query analysis for demo
        self.mock_query_analysis = QueryAnalysis(
            original_query="What is the budget allocation for Q4?",
            intent=QueryIntent(name="FACTUAL", confidence=0.9, keywords=["budget", "allocation", "Q4"]),
            entities=[
                QueryEntity(text="budget", category="FINANCIAL", confidence=0.95),
                QueryEntity(text="Q4", category="TIME", confidence=0.85),
                QueryEntity(text="allocation", category="PROCESS", confidence=0.90)
            ],
            complexity=QueryComplexity.MEDIUM,
            suggested_chunks=10,
            processing_time_ms=150.0,
            confidence_score=0.88
        )
    
    async def demo_streaming_modes(self):
        """Demonstrate different streaming modes"""
        
        print("🎯 Modern RAG App - Streaming Demo")
        print("=" * 50)
        
        test_queries = [
            "What is the budget allocation for Q4?",
            "How does our spending compare to last year?", 
            "What are the key financial priorities?"
        ]
        
        modes = [
            ("Speed Mode", SummarizationMode.FAST, 30, 30),
            ("Balanced Mode", SummarizationMode.BALANCED, 50, 50),
            ("Comprehensive Mode", SummarizationMode.COMPREHENSIVE, 70, 70)
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 40)
            
            for mode_name, mode, chunk_size, delay in modes:
                print(f"\n🚀 {mode_name} (chunks: {chunk_size}, delay: {delay}ms)")
                await self.demo_streaming_response(query, mode, chunk_size, delay)
                
                # Short pause between modes
                await asyncio.sleep(1)
    
    async def demo_streaming_response(self, 
                                    query: str, 
                                    mode: SummarizationMode,
                                    chunk_size: int,
                                    delay_ms: int):
        """Demonstrate streaming response for a single query"""
        
        try:
            # Create streaming config
            config = create_streaming_summarization_config(
                mode=mode,
                chunk_size=chunk_size,
                delay_ms=delay_ms
            )
            
            # Initialize summarizer
            summarizer = IntelligentSummarizer()
            
            # Mock retrieval result for demo
            from models.retrieval_models import RetrievalResult, RankedChunk
            
            mock_chunks = [
                RankedChunk(
                    content="The Q4 budget shows an allocation of $2.5M for operational expenses, with $800K dedicated to marketing initiatives and $1.2M for development projects.",
                    source="budget_q4_2024.pdf", 
                    relevance_score=0.95,
                    chunk_index=1
                ),
                RankedChunk(
                    content="Compared to Q3, we see a 15% increase in allocated funds for technology infrastructure, reflecting our strategic focus on digital transformation.",
                    source="budget_comparison.pdf",
                    relevance_score=0.87,
                    chunk_index=2
                )
            ]
            
            mock_retrieval = RetrievalResult(
                query="budget allocation Q4",
                ranked_chunks=mock_chunks,
                total_chunks_found=2,
                processing_time_ms=250.0,
                overall_precision=0.91,
                retrieval_strategy="hybrid"
            )
            
            # Stream the response
            response_stream = await summarizer.summarize_with_context(
                query, self.mock_query_analysis, mock_retrieval, config
            )
            
            # Display streaming output
            full_response = ""
            chunk_count = 0
            
            async for chunk in response_stream:
                chunk_count += 1
                
                if chunk.metadata and chunk.metadata.get("type") == "initialization":
                    print(f"🔄 Starting: {chunk.metadata.get('processing_mode', 'unknown')}")
                elif chunk.metadata and chunk.metadata.get("type") == "content":
                    print(f"📤 Chunk {chunk.chunk_id}: {chunk.content}", end='', flush=True)
                    full_response += chunk.content
                elif chunk.metadata and chunk.metadata.get("type") == "completion":
                    print(f"\n✅ Complete: {chunk.metadata.get('total_chunks', 0)} chunks")
                    break
                elif chunk.metadata and chunk.metadata.get("type") == "error":
                    print(f"\n❌ Error: {chunk.metadata.get('error')}")
                    break
                
                # Simulate real-time display
                await asyncio.sleep(delay_ms / 1000.0)
            
            print(f"\n📊 Response Statistics:")
            print(f"   - Total chunks: {chunk_count}")
            print(f"   - Response length: {len(full_response)} characters")
            print(f"   - Mode: {mode.value}")
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
    
    async def demo_api_streaming(self):
        """Demonstrate API-style streaming interaction"""
        
        print("\n🌐 API Streaming Demo")
        print("=" * 30)
        
        # Simulate API request
        request_data = {
            "query": "What are the key budget highlights?",
            "mode": "balanced",
            "stream_chunk_size": 40,
            "stream_delay_ms": 60,
            "collection_name": "demo"
        }
        
        print(f"📡 Request: {json.dumps(request_data, indent=2)}")
        print("\n📺 Streaming Response:")
        print("-" * 25)
        
        # Simulate Server-Sent Events format
        events = [
            {"type": "chunk", "chunk_id": 0, "content": "", "metadata": {"type": "initialization", "stage": "starting"}},
            {"type": "chunk", "chunk_id": 1, "content": "", "metadata": {"stage": "analysis", "message": "Analyzing query..."}},
            {"type": "chunk", "chunk_id": 2, "content": "", "metadata": {"stage": "retrieval", "message": "Finding relevant content..."}},
            {"type": "chunk", "chunk_id": 3, "content": "", "metadata": {"stage": "generation", "message": "Generating response..."}},
            {"type": "chunk", "chunk_id": 4, "content": "Based on the budget analysis", "metadata": {"type": "content", "progress": 0.1}},
            {"type": "chunk", "chunk_id": 5, "content": ", the key highlights include", "metadata": {"type": "content", "progress": 0.2}},
            {"type": "chunk", "chunk_id": 6, "content": " a 12% increase in operational", "metadata": {"type": "content", "progress": 0.4}},
            {"type": "chunk", "chunk_id": 7, "content": " funding and strategic investments", "metadata": {"type": "content", "progress": 0.6}},
            {"type": "chunk", "chunk_id": 8, "content": " in technology infrastructure", "metadata": {"type": "content", "progress": 0.8}},
            {"type": "chunk", "chunk_id": 9, "content": " totaling $3.2M for the quarter.", "metadata": {"type": "content", "progress": 1.0}},
            {"type": "final", "completed": True, "response": {"confidence_score": 0.89, "sources_used": ["budget_2024.pdf"]}}
        ]
        
        for event in events:
            print(f"data: {json.dumps(event)}")
            
            if event["type"] == "chunk":
                if event.get("metadata", {}).get("stage"):
                    print(f"      → Status: {event['metadata']['message']}")
                elif event.get("metadata", {}).get("type") == "content":
                    print(f"      → Content: {event['content']}")
                    print(f"      → Progress: {event['metadata']['progress']:.1%}")
            elif event["type"] == "final":
                print(f"      → Complete: Confidence {event['response']['confidence_score']:.2f}")
            
            await asyncio.sleep(0.1)  # Simulate streaming delay
        
        print("\n✅ Streaming demo complete!")
    
    async def run_full_demo(self):
        """Run complete streaming demonstration"""
        
        print("🎬 Starting Modern RAG Streaming Demo...")
        
        # Demo 1: Different streaming modes
        await self.demo_streaming_modes()
        
        # Demo 2: API-style streaming
        await self.demo_api_streaming()
        
        print(f"\n🎉 Demo Complete!")
        print("=" * 50)
        print("Key Features Demonstrated:")
        print("✓ Multiple streaming modes (Fast/Balanced/Comprehensive)")
        print("✓ Configurable chunk sizes and delays")
        print("✓ Real-time status updates")
        print("✓ Progressive content delivery")
        print("✓ Server-Sent Events format")
        print("✓ Error handling and completion")
        print("\n💡 Next Steps:")
        print("- Start the API server: python api/v1/endpoints.py")
        print("- Test streaming: POST /ask/stream")
        print("- Access docs: http://localhost:8000/docs")


# Run demo if called directly
if __name__ == "__main__":
    async def main():
        demo = StreamingDemo()
        await demo.run_full_demo()
    
    # Run the demo
    asyncio.run(main())
