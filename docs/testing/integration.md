# Integration Testing Guide

## 📋 Overview

Integration testing validates the interaction between different components of the Modern RAG Application. These tests ensure that services work together correctly, data flows properly through the system, and the complete RAG pipeline delivers expected results.

## 🏗️ Integration Test Architecture

### **Test Categories**
1. **Service Integration** - Inter-service communication
2. **Data Flow Integration** - End-to-end data processing
3. **API Integration** - Frontend-backend integration
4. **Storage Integration** - Database and vector store integration
5. **External Service Integration** - Third-party service integration

### **Test Environment Setup**
```python
# tests/test_integration/conftest.py
import pytest
import asyncio
from pathlib import Path
import tempfile

from modern_rag_app.app import create_app
from modern_rag_app.config.settings import Settings
from modern_rag_app.services.rag_orchestrator import RAGOrchestrator
from modern_rag_app.storage.chroma_store import ChromaStore

@pytest.fixture(scope="session")
async def integration_app():
    """Create application for integration testing."""
    settings = Settings(
        environment="integration",
        database_url="sqlite:///integration_test.db",
        chroma_persist_directory=tempfile.mkdtemp(),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        log_level="INFO"
    )
    
    app = create_app(settings)
    await app.startup()
    
    yield app
    
    await app.shutdown()

@pytest.fixture(scope="function")
async def rag_orchestrator(integration_app):
    """RAG orchestrator with all services initialized."""
    from modern_rag_app.services import get_rag_orchestrator
    
    orchestrator = get_rag_orchestrator()
    await orchestrator.initialize()
    
    yield orchestrator
    
    await orchestrator.cleanup()

@pytest.fixture
def sample_documents():
    """Comprehensive sample documents for integration testing."""
    return [
        {
            "id": "ai_overview",
            "content": """
            Artificial Intelligence (AI) is a broad field of computer science focused on 
            creating systems capable of performing tasks that typically require human 
            intelligence. This includes machine learning, natural language processing, 
            computer vision, and robotics. Modern AI systems use deep learning and 
            neural networks to achieve remarkable performance in various domains.
            """,
            "metadata": {
                "source": "AI_Overview.pdf",
                "page": 1,
                "title": "Introduction to Artificial Intelligence",
                "category": "overview",
                "created_date": "2024-01-15"
            }
        },
        {
            "id": "ml_fundamentals", 
            "content": """
            Machine Learning (ML) is a subset of artificial intelligence that enables 
            computers to learn and improve from experience without being explicitly 
            programmed. ML algorithms build mathematical models based on training data 
            to make predictions or decisions. The three main types are supervised learning, 
            unsupervised learning, and reinforcement learning.
            """,
            "metadata": {
                "source": "ML_Fundamentals.pdf",
                "page": 3,
                "title": "Machine Learning Fundamentals", 
                "category": "technical",
                "created_date": "2024-01-20"
            }
        },
        {
            "id": "neural_networks",
            "content": """
            Neural networks are computing systems inspired by biological neural networks. 
            They consist of interconnected nodes (neurons) organized in layers. Deep 
            neural networks with multiple hidden layers have revolutionized fields like 
            image recognition, natural language processing, and game playing. Popular 
            architectures include CNNs, RNNs, and Transformers.
            """,
            "metadata": {
                "source": "Neural_Networks.pdf", 
                "page": 7,
                "title": "Neural Network Architectures",
                "category": "technical",
                "created_date": "2024-01-25"
            }
        }
    ]
```

## 🔄 Service Integration Tests

### **RAG Pipeline Integration**
```python
class TestRAGPipelineIntegration:
    """Test complete RAG pipeline integration."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_rag_flow(self, rag_orchestrator, sample_documents):
        """Test complete document ingestion and query flow."""
        
        # Phase 1: Document Ingestion
        print("Phase 1: Document Ingestion")
        ingestion_result = await rag_orchestrator.ingest_documents(
            documents=sample_documents,
            processing_config={
                "chunk_size": 512,
                "chunk_overlap": 50,
                "enable_metadata_extraction": True,
                "quality_threshold": 0.7
            }
        )
        
        # Validate ingestion
        assert ingestion_result.success
        assert ingestion_result.processed_count == len(sample_documents)
        assert ingestion_result.chunks_created > 0
        assert len(ingestion_result.failed_documents) == 0
        
        print(f"Processed {ingestion_result.processed_count} documents")
        print(f"Created {ingestion_result.chunks_created} chunks")
        
        # Phase 2: Query Processing
        print("\nPhase 2: Query Processing")
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning types", 
            "How do neural networks work?",
            "Compare supervised and unsupervised learning"
        ]
        
        for query in test_queries:
            print(f"Testing query: {query}")
            
            response = await rag_orchestrator.process_query(
                query=query,
                config={
                    "performance_mode": "balanced",
                    "max_results": 5,
                    "enable_reranking": True,
                    "min_relevance_score": 0.3
                }
            )
            
            # Validate response
            assert response.success
            assert response.answer is not None
            assert len(response.answer) > 0
            assert len(response.sources) > 0
            assert response.confidence_score > 0.5
            assert response.processing_time < 2.0
            
            # Validate sources
            for source in response.sources:
                assert source.relevance_score > 0.3
                assert source.content is not None
                assert source.metadata is not None
            
            print(f"  Answer length: {len(response.answer)} chars")
            print(f"  Sources found: {len(response.sources)}")
            print(f"  Confidence: {response.confidence_score:.3f}")
            print(f"  Processing time: {response.processing_time:.3f}s")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_analysis_integration(self, rag_orchestrator):
        """Test query analysis integration with retrieval."""
        
        complex_queries = [
            {
                "query": "What are the differences between CNNs and RNNs in deep learning?",
                "expected_intent": "comparative",
                "expected_entities": ["CNN", "RNN", "deep learning"]
            },
            {
                "query": "How can I implement a basic neural network?",
                "expected_intent": "how_to", 
                "expected_entities": ["neural network", "implementation"]
            },
            {
                "query": "Show me examples of supervised learning algorithms",
                "expected_intent": "example",
                "expected_entities": ["supervised learning", "algorithms"]
            }
        ]
        
        for test_case in complex_queries:
            query = test_case["query"]
            
            # Analyze query first
            analysis = await rag_orchestrator.query_analyzer.analyze_query(query)
            
            # Validate analysis
            assert analysis.intent.value == test_case["expected_intent"]
            
            entity_texts = [e.text.lower() for e in analysis.entities]
            for expected_entity in test_case["expected_entities"]:
                assert any(expected_entity.lower() in text for text in entity_texts)
            
            # Process full query with analysis
            response = await rag_orchestrator.process_query(query)
            
            # Validate response incorporates analysis
            assert response.success
            assert len(response.sources) > 0
            
            # Check that retrieved sources are relevant to analyzed entities
            source_content = " ".join(source.content.lower() for source in response.sources)
            for expected_entity in test_case["expected_entities"]:
                assert expected_entity.lower() in source_content
```

### **Storage Integration Tests**
```python
class TestStorageIntegration:
    """Test storage layer integration."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_vector_store_integration(self, rag_orchestrator, sample_documents):
        """Test vector store operations integration."""
        
        # Ingest documents
        await rag_orchestrator.ingest_documents(sample_documents)
        
        # Test direct vector store operations
        vector_store = rag_orchestrator.vector_store
        
        # Test similarity search
        query_embedding = await rag_orchestrator.embedding_service.generate_embeddings(
            ["artificial intelligence"]
        )
        
        results = await vector_store.similarity_search(
            query_embedding=query_embedding[0],
            k=5,
            score_threshold=0.5
        )
        
        assert len(results) > 0
        for result in results:
            assert result.similarity_score >= 0.5
            assert result.content is not None
            assert result.metadata is not None
        
        # Test filtered search
        filtered_results = await vector_store.similarity_search(
            query_embedding=query_embedding[0],
            k=3,
            filters={"category": "technical"}
        )
        
        for result in filtered_results:
            assert result.metadata.get("category") == "technical"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_metadata_filtering_integration(self, rag_orchestrator, sample_documents):
        """Test metadata filtering across the pipeline."""
        
        # Ingest documents with rich metadata
        await rag_orchestrator.ingest_documents(sample_documents)
        
        # Test category filtering
        response = await rag_orchestrator.process_query(
            query="machine learning concepts",
            config={
                "filters": {"category": "technical"},
                "max_results": 10
            }
        )
        
        assert response.success
        assert len(response.sources) > 0
        
        # Verify all sources match filter
        for source in response.sources:
            assert source.metadata.get("category") == "technical"
        
        # Test date range filtering
        response = await rag_orchestrator.process_query(
            query="AI overview",
            config={
                "filters": {
                    "created_date": {"gte": "2024-01-01", "lte": "2024-01-31"}
                }
            }
        )
        
        assert response.success
        
        # Test source filtering
        response = await rag_orchestrator.process_query(
            query="neural networks",
            config={
                "filters": {"source": "Neural_Networks.pdf"}
            }
        )
        
        assert response.success
        for source in response.sources:
            assert "Neural_Networks.pdf" in source.metadata.get("source", "")
```

## 🌐 API Integration Tests

### **REST API Integration**
```python
class TestAPIIntegration:
    """Test API integration with backend services."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_api_integration(self, authenticated_client, sample_documents):
        """Test query API integration with RAG pipeline."""
        
        # First, ingest documents via API
        for doc in sample_documents:
            response = await authenticated_client.post(
                "/api/v1/documents/ingest",
                json={
                    "document": doc,
                    "processing_config": {
                        "chunk_size": 512,
                        "enable_metadata_extraction": True
                    }
                }
            )
            assert response.status_code == 200
            
            result = response.json()
            assert result["success"] is True
        
        # Test query processing via API
        query_request = {
            "query": "What is machine learning?",
            "config": {
                "performance_mode": "balanced",
                "max_results": 5,
                "enable_reranking": True
            }
        }
        
        response = await authenticated_client.post(
            "/api/v1/query",
            json=query_request
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence_score" in data
        assert "processing_time" in data
        
        # Validate response structure
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0
        assert 0.0 <= data["confidence_score"] <= 1.0
        assert data["processing_time"] > 0
        
        # Validate source structure
        for source in data["sources"]:
            assert "content" in source
            assert "relevance_score" in source
            assert "metadata" in source
            assert 0.0 <= source["relevance_score"] <= 1.0
    
    @pytest.mark.integration
    @pytest.mark.asyncio 
    async def test_streaming_api_integration(self, authenticated_client):
        """Test streaming API integration."""
        
        query_request = {
            "query": "Explain artificial intelligence and its applications",
            "config": {
                "enable_streaming": True,
                "performance_mode": "accuracy"
            }
        }
        
        async with authenticated_client.stream(
            "POST", "/api/v1/query/stream", json=query_request
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
            
            events = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event_data = line[6:]  # Remove "data: " prefix
                    if event_data.strip():
                        events.append(json.loads(event_data))
            
            # Validate streaming events
            assert len(events) > 0
            
            event_types = [event.get("type") for event in events]
            assert "partial" in event_types
            assert "sources" in event_types  
            assert "complete" in event_types
            
            # Validate final complete event
            complete_event = next(e for e in events if e.get("type") == "complete")
            assert "answer" in complete_event
            assert "sources" in complete_event
            assert "confidence_score" in complete_event
```

### **WebSocket Integration**
```python
class TestWebSocketIntegration:
    """Test WebSocket integration for real-time features."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_query_processing(self, test_app):
        """Test real-time query processing via WebSocket."""
        
        from httpx import AsyncClient
        
        async with AsyncClient(app=test_app) as client:
            async with client.websocket_connect("/ws/query") as websocket:
                
                # Send query
                query_message = {
                    "type": "query",
                    "query": "What are neural networks?",
                    "config": {
                        "performance_mode": "balanced",
                        "enable_streaming": True
                    }
                }
                
                await websocket.send_json(query_message)
                
                # Collect responses
                responses = []
                while True:
                    try:
                        message = await asyncio.wait_for(
                            websocket.receive_json(), timeout=5.0
                        )
                        responses.append(message)
                        
                        if message.get("type") == "complete":
                            break
                            
                    except asyncio.TimeoutError:
                        break
                
                # Validate responses
                assert len(responses) > 0
                
                message_types = [msg.get("type") for msg in responses]
                assert "partial" in message_types
                assert "complete" in message_types
                
                # Validate final response
                final_response = next(msg for msg in responses if msg.get("type") == "complete")
                assert "answer" in final_response
                assert "sources" in final_response
                assert final_response.get("success") is True
```

## 📊 Performance Integration Tests

### **Load Testing Integration**
```python
class TestPerformanceIntegration:
    """Test performance under realistic load conditions."""
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_user_simulation(self, rag_orchestrator, sample_documents):
        """Simulate multiple concurrent users."""
        
        # Setup: Ingest documents
        await rag_orchestrator.ingest_documents(sample_documents)
        
        # Define user scenarios
        user_queries = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "How do neural networks work?",
            "Types of AI algorithms",
            "Deep learning applications"
        ]
        
        async def simulate_user(user_id: int):
            """Simulate a single user session."""
            results = []
            
            for i, query in enumerate(user_queries):
                start_time = time.time()
                
                try:
                    response = await rag_orchestrator.process_query(
                        query=query,
                        config={"performance_mode": "balanced"}
                    )
                    
                    end_time = time.time()
                    
                    results.append({
                        "user_id": user_id,
                        "query_index": i,
                        "success": response.success,
                        "response_time": end_time - start_time,
                        "confidence": response.confidence_score,
                        "sources_count": len(response.sources)
                    })
                    
                    # Simulate user think time
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    results.append({
                        "user_id": user_id,
                        "query_index": i,
                        "success": False,
                        "error": str(e),
                        "response_time": time.time() - start_time
                    })
            
            return results
        
        # Run concurrent users
        num_users = 10
        start_time = time.time()
        
        user_tasks = [simulate_user(i) for i in range(num_users)]
        all_results = await asyncio.gather(*user_tasks)
        
        end_time = time.time()
        total_test_time = end_time - start_time
        
        # Analyze results
        flat_results = [result for user_results in all_results for result in user_results]
        
        success_rate = sum(1 for r in flat_results if r["success"]) / len(flat_results)
        avg_response_time = np.mean([r["response_time"] for r in flat_results if r["success"]])
        p95_response_time = np.percentile([r["response_time"] for r in flat_results if r["success"]], 95)
        
        # Performance assertions
        assert success_rate >= 0.95  # 95% success rate
        assert avg_response_time < 1.0  # Average under 1 second
        assert p95_response_time < 2.0  # 95th percentile under 2 seconds
        
        print(f"Concurrent users: {num_users}")
        print(f"Total test time: {total_test_time:.2f}s")
        print(f"Success rate: {success_rate:.3f}")
        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"95th percentile response time: {p95_response_time:.3f}s")
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_integration(self, rag_orchestrator):
        """Test memory usage under sustained load."""
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large document set
        large_documents = []
        for i in range(50):
            large_documents.append({
                "id": f"large_doc_{i}",
                "content": "Large document content. " * 1000,  # ~25KB per doc
                "metadata": {"source": f"large_file_{i}.pdf", "page": i}
            })
        
        # Ingest documents
        await rag_orchestrator.ingest_documents(large_documents)
        
        # Process many queries
        for i in range(100):
            query = f"Test query {i} about large documents"
            await rag_orchestrator.process_query(query)
            
            if i % 10 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory should not grow excessively
                assert memory_increase < 500  # Less than 500MB increase
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Total increase: {total_increase:.1f} MB")
        
        assert total_increase < 1000  # Less than 1GB total increase
```

## 🔧 Test Data Management

### **Test Data Persistence**
```python
class TestDataPersistence:
    """Test data persistence across service restarts."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_persistence_integration(self, integration_app, sample_documents):
        """Test that data persists across service restarts."""
        
        # Phase 1: Initial setup and data ingestion
        orchestrator1 = await get_rag_orchestrator()
        await orchestrator1.initialize()
        
        ingestion_result = await orchestrator1.ingest_documents(sample_documents)
        assert ingestion_result.success
        
        # Verify data exists
        response1 = await orchestrator1.process_query("artificial intelligence")
        assert response1.success
        assert len(response1.sources) > 0
        
        # Store reference data
        original_sources = response1.sources
        
        await orchestrator1.cleanup()
        
        # Phase 2: Restart services
        orchestrator2 = await get_rag_orchestrator()
        await orchestrator2.initialize()
        
        # Verify data still exists after restart
        response2 = await orchestrator2.process_query("artificial intelligence")
        assert response2.success
        assert len(response2.sources) > 0
        
        # Compare results
        assert len(response2.sources) == len(original_sources)
        
        # Verify source content matches
        source_contents_1 = {src.content for src in original_sources}
        source_contents_2 = {src.content for src in response2.sources}
        assert source_contents_1 == source_contents_2
        
        await orchestrator2.cleanup()
```

## 🚨 Error Recovery Integration Tests

### **Failure Recovery**
```python
class TestErrorRecoveryIntegration:
    """Test error recovery and system resilience."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_service_failure_recovery(self, rag_orchestrator):
        """Test system behavior when individual services fail."""
        
        # Test embedding service failure
        with patch.object(rag_orchestrator.embedding_service, 'generate_embeddings') as mock_embed:
            mock_embed.side_effect = Exception("Embedding service down")
            
            # System should handle gracefully with fallback
            response = await rag_orchestrator.process_query(
                query="test query",
                config={"enable_fallback": True}
            )
            
            # Should not crash, may use cached results or keyword search
            assert response is not None
            if not response.success:
                assert "embedding" in response.error_message.lower()
        
        # Test vector store failure
        with patch.object(rag_orchestrator.vector_store, 'similarity_search') as mock_search:
            mock_search.side_effect = Exception("Vector store unavailable")
            
            response = await rag_orchestrator.process_query(
                query="test query",
                config={"enable_fallback": True}
            )
            
            # Should handle gracefully
            assert response is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_partial_service_degradation(self, rag_orchestrator):
        """Test system behavior under partial service degradation."""
        
        # Simulate slow response times
        async def slow_embedding(*args, **kwargs):
            await asyncio.sleep(2.0)  # 2 second delay
            return [[0.1] * 384]  # Return dummy embedding
        
        with patch.object(rag_orchestrator.embedding_service, 'generate_embeddings', slow_embedding):
            
            start_time = time.time()
            response = await rag_orchestrator.process_query(
                query="test query",
                config={"max_response_time": 1.0}  # 1 second timeout
            )
            end_time = time.time()
            
            # Should timeout gracefully
            assert end_time - start_time < 1.5  # Allow some overhead
            
            if not response.success:
                assert "timeout" in response.error_message.lower()
```

## 📈 Monitoring Integration

### **Health Check Integration**
```python
class TestHealthCheckIntegration:
    """Test health monitoring integration."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_system_health_checks(self, rag_orchestrator):
        """Test comprehensive system health checks."""
        
        health_status = await rag_orchestrator.get_health_status()
        
        # Validate overall health
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]
        assert "services" in health_status
        assert "performance" in health_status
        assert "storage" in health_status
        
        # Validate service health
        services = health_status["services"]
        required_services = [
            "query_analyzer", "embedding_service", "vector_store", 
            "text_processor", "chunking_service"
        ]
        
        for service in required_services:
            assert service in services
            assert services[service]["status"] in ["healthy", "degraded", "unhealthy"]
            assert "response_time" in services[service]
        
        # Validate performance metrics
        performance = health_status["performance"]
        assert "average_query_time" in performance
        assert "queries_per_minute" in performance
        assert "error_rate" in performance
        
        # Performance thresholds
        assert performance["average_query_time"] < 2.0
        assert performance["error_rate"] < 0.05  # Less than 5%
```

## 🚀 Running Integration Tests

### **Test Execution Commands**
```bash
# Run all integration tests
pytest tests/test_integration/ -m integration

# Run specific integration test categories
pytest tests/test_integration/ -m "integration and not performance"
pytest tests/test_integration/ -m "integration and performance"

# Run with detailed output
pytest tests/test_integration/ -v -s --tb=short

# Run integration tests with coverage
pytest tests/test_integration/ --cov=modern_rag_app --cov-report=html

# Run integration tests in parallel (be careful with shared resources)
pytest tests/test_integration/ -n 2  # Use 2 processes max for integration
```

### **Environment Setup**
```bash
# Set up integration test environment
export ENVIRONMENT=integration
export LOG_LEVEL=INFO
export CHROMA_PERSIST_DIRECTORY=/tmp/integration_chroma
export DATABASE_URL=sqlite:///integration_test.db

# Run integration tests
pytest tests/test_integration/
```

## 📚 Related Documentation

- **[Testing Overview](README.md)** - Main testing documentation
- **[API Testing](../api/testing.md)** - API-specific tests
- **[Performance Testing](performance.md)** - Performance benchmarks
- **[Services Documentation](../services/)** - Individual service details

---

**Integration testing ensures all components work together seamlessly in the Modern RAG Application.**
