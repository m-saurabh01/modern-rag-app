# Testing Documentation

## 📋 Overview

The Modern RAG Application includes comprehensive testing infrastructure covering unit tests, integration tests, API testing, performance benchmarks, and end-to-end validation. The testing framework ensures reliability, performance, and correctness across all system components.

## 🏗️ Testing Architecture

### **Test Structure**
```
tests/
├── __init__.py                 # Testing utilities and fixtures
├── conftest.py                # Pytest configuration and shared fixtures
├── test_requirements.txt      # Testing dependencies
├── test_api/                  # API endpoint testing
│   ├── __init__.py
│   ├── test_endpoints.py      # REST API tests
│   ├── test_websocket.py      # WebSocket API tests
│   ├── test_authentication.py # Auth testing
│   └── test_error_handling.py # Error response tests
├── test_integration/          # Cross-component integration tests
│   ├── __init__.py
│   ├── test_rag_pipeline.py   # End-to-end RAG tests
│   ├── test_document_flow.py  # Document processing flow
│   ├── test_query_flow.py     # Query processing flow
│   └── test_performance.py    # Performance integration tests
├── test_processing/           # Document processing tests
│   ├── __init__.py
│   ├── test_pdf_processor.py  # PDF processing tests
│   ├── test_ocr_processor.py  # OCR functionality tests
│   └── test_chunking.py       # Text chunking tests
└── test_services/             # Service layer tests
    ├── __init__.py
    ├── test_query_analyzer.py     # Query analysis tests
    ├── test_intelligent_retriever.py # Retrieval tests
    ├── test_rag_orchestrator.py   # Orchestrator tests
    ├── test_embedding_service.py  # Embedding tests
    ├── test_chunking_service.py   # Chunking service tests
    └── test_text_processor.py     # Text processing tests
```

## 🧪 Testing Categories

### **1. Unit Tests**
Individual component testing with mocked dependencies:

```python
# Example: Query Analyzer Unit Test
import pytest
from unittest.mock import Mock, patch
from services.query_analyzer import QueryAnalyzer
from models.query_models import QueryAnalysis, QueryIntent

class TestQueryAnalyzer:
    """Unit tests for query analyzer service."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance with mocked dependencies."""
        return QueryAnalyzer(
            intent_classifier=Mock(),
            entity_extractor=Mock(),
            similarity_analyzer=Mock()
        )
    
    @pytest.mark.asyncio
    async def test_simple_query_analysis(self, analyzer):
        """Test analysis of simple factual query."""
        query = "What is machine learning?"
        
        # Mock intent classification
        analyzer.intent_classifier.classify.return_value = QueryIntent.FACTUAL
        
        # Mock entity extraction
        analyzer.entity_extractor.extract.return_value = [
            EntityResult(text="machine learning", type="CONCEPT", confidence=0.9)
        ]
        
        # Perform analysis
        result = await analyzer.analyze_query(query)
        
        # Assertions
        assert isinstance(result, QueryAnalysis)
        assert result.intent == QueryIntent.FACTUAL
        assert len(result.entities) == 1
        assert result.entities[0].text == "machine learning"
        assert result.complexity == QueryComplexity.SIMPLE
```

### **2. Integration Tests**
Cross-component interaction testing:

```python
# Example: RAG Pipeline Integration Test
import pytest
from tests.fixtures import sample_documents, test_database
from services.rag_orchestrator import RAGOrchestrator

class TestRAGPipelineIntegration:
    """Integration tests for complete RAG pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_rag_flow(self, rag_orchestrator, sample_documents):
        """Test complete document ingestion and retrieval flow."""
        
        # 1. Document ingestion
        ingestion_result = await rag_orchestrator.ingest_documents(
            documents=sample_documents,
            processing_config={"enable_ocr": True, "chunk_size": 512}
        )
        
        assert ingestion_result.success
        assert ingestion_result.processed_count == len(sample_documents)
        
        # 2. Query processing and retrieval
        query = "What are the key findings about neural networks?"
        response = await rag_orchestrator.process_query(
            query=query,
            performance_mode="balanced"
        )
        
        # Assertions
        assert response.success
        assert len(response.sources) > 0
        assert response.confidence_score > 0.7
        assert "neural networks" in response.answer.lower()
        
        # 3. Validate retrieval quality
        for source in response.sources:
            assert source.relevance_score > 0.5
            assert source.content is not None
```

### **3. API Tests**
REST and WebSocket endpoint testing:

```python
# Example: API Endpoint Tests
import pytest
from httpx import AsyncClient
from tests.fixtures import test_app, authenticated_client

class TestAPIEndpoints:
    """API endpoint tests."""
    
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_query_endpoint(self, authenticated_client: AsyncClient):
        """Test query processing endpoint."""
        
        query_data = {
            "query": "Explain deep learning",
            "config": {
                "performance_mode": "balanced",
                "max_results": 10,
                "enable_streaming": False
            }
        }
        
        response = await authenticated_client.post(
            "/api/v1/query",
            json=query_data
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence_score" in data
        assert data["confidence_score"] >= 0.0
        assert data["confidence_score"] <= 1.0
    
    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_streaming_query(self, authenticated_client: AsyncClient):
        """Test streaming query endpoint."""
        
        query_data = {
            "query": "Comprehensive analysis of AI trends",
            "config": {"enable_streaming": True}
        }
        
        async with authenticated_client.stream(
            "POST", "/api/v1/query/stream", json=query_data
        ) as response:
            assert response.status_code == 200
            
            chunks = []
            async for chunk in response.aiter_text():
                if chunk.strip():
                    chunks.append(chunk)
            
            assert len(chunks) > 0
            # Validate streaming format
            for chunk in chunks[:-1]:  # All but last should be partial
                data = json.loads(chunk)
                assert data["type"] in ["partial", "sources", "complete"]
```

## 🔧 Test Configuration

### **pytest.ini**
```ini
[tool:pytest]
minversion = 6.0
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=modern_rag_app
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, database required)
    api: API endpoint tests
    performance: Performance benchmarks
    slow: Slow tests (> 5 seconds)
    gpu: GPU-dependent tests
    external: Tests requiring external services
```

### **conftest.py - Shared Fixtures**
```python
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock

from modern_rag_app.app import create_app
from modern_rag_app.config.settings import Settings
from modern_rag_app.storage.chroma_store import ChromaStore

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_settings():
    """Test application settings."""
    return Settings(
        environment="testing",
        database_url="sqlite:///test.db",
        chroma_persist_directory=tempfile.mkdtemp(),
        log_level="DEBUG",
        enable_cors=True
    )

@pytest.fixture(scope="session")
async def test_app(test_settings):
    """Create test application instance."""
    app = create_app(test_settings)
    yield app

@pytest.fixture(scope="function")
async def test_database(test_settings):
    """Create clean test database for each test."""
    db_path = Path(test_settings.database_url.replace("sqlite:///", ""))
    
    # Setup
    db_path.parent.mkdir(exist_ok=True)
    
    yield test_settings.database_url
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()

@pytest.fixture(scope="function")
async def chroma_store(test_settings):
    """Create clean ChromaDB instance for each test."""
    store = ChromaStore(
        persist_directory=test_settings.chroma_persist_directory,
        collection_name="test_collection"
    )
    
    await store.initialize()
    yield store
    
    # Cleanup
    await store.clear_collection()

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "Machine learning is a subset of artificial intelligence...",
            "metadata": {"source": "ML_basics.pdf", "page": 1}
        },
        {
            "id": "doc2", 
            "content": "Neural networks are computing systems inspired by biological neural networks...",
            "metadata": {"source": "Neural_Networks.pdf", "page": 1}
        }
    ]

@pytest.fixture
async def authenticated_client(test_app):
    """Authenticated HTTP client for API tests."""
    from httpx import AsyncClient
    
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        # Add authentication if required
        # client.headers["Authorization"] = "Bearer test_token"
        yield client
```

## 📊 Performance Testing

### **Performance Benchmarks**
```python
import pytest
import time
import asyncio
from statistics import mean, stdev

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_query_response_time(self, rag_orchestrator):
        """Test query response time benchmarks."""
        
        queries = [
            "What is machine learning?",
            "Explain neural network architectures",
            "Compare supervised and unsupervised learning",
            "Describe deep learning applications"
        ]
        
        response_times = []
        
        for query in queries:
            start_time = time.time()
            
            response = await rag_orchestrator.process_query(
                query=query,
                performance_mode="speed"
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Assert response quality
            assert response.success
            assert response.confidence_score > 0.5
        
        # Performance assertions
        avg_response_time = mean(response_times)
        max_response_time = max(response_times)
        
        assert avg_response_time < 0.5  # 500ms average
        assert max_response_time < 1.0  # 1s maximum
        
        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"Max response time: {max_response_time:.3f}s")
        print(f"Standard deviation: {stdev(response_times):.3f}s")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, rag_orchestrator):
        """Test concurrent query handling."""
        
        async def single_query(query_id: int):
            query = f"Test query {query_id} about artificial intelligence"
            start_time = time.time()
            
            response = await rag_orchestrator.process_query(query)
            
            end_time = time.time()
            return {
                "query_id": query_id,
                "response_time": end_time - start_time,
                "success": response.success,
                "confidence": response.confidence_score
            }
        
        # Run 10 concurrent queries
        concurrent_queries = 10
        tasks = [single_query(i) for i in range(concurrent_queries)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Assertions
        assert all(r["success"] for r in results)
        assert total_time < 2.0  # All queries should complete in 2s
        
        avg_response_time = mean(r["response_time"] for r in results)
        assert avg_response_time < 1.0  # Average should be under 1s
        
        print(f"Concurrent queries: {concurrent_queries}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average response time: {avg_response_time:.3f}s")
```

## 🔍 Test Data Management

### **Test Data Fixtures**
```python
@pytest.fixture(scope="session")
def test_documents_large():
    """Large document set for comprehensive testing."""
    documents = []
    
    # Generate test documents
    for i in range(100):
        documents.append({
            "id": f"test_doc_{i}",
            "content": f"This is test document {i} containing information about "
                      f"topic {i % 10}. It includes detailed analysis and examples.",
            "metadata": {
                "source": f"test_file_{i}.pdf",
                "page": i % 50 + 1,
                "category": f"category_{i % 5}",
                "created_date": "2024-01-01"
            }
        })
    
    return documents

@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    mock = Mock()
    
    # Mock embedding generation
    mock.generate_embeddings.return_value = [
        [0.1, 0.2, 0.3] * 128  # 384-dimensional embedding
        for _ in range(10)
    ]
    
    mock.embedding_dimension = 384
    mock.model_name = "test-embedding-model"
    
    return mock
```

## 🚨 Error Testing

### **Error Handling Tests**
```python
class TestErrorHandling:
    """Error handling and edge case tests."""
    
    @pytest.mark.asyncio
    async def test_invalid_query_handling(self, rag_orchestrator):
        """Test handling of invalid queries."""
        
        invalid_queries = [
            "",  # Empty query
            " " * 1000,  # Very long whitespace
            "?" * 500,  # Excessive punctuation
            None  # None value
        ]
        
        for query in invalid_queries:
            with pytest.raises(ValueError):
                await rag_orchestrator.process_query(query)
    
    @pytest.mark.asyncio
    async def test_service_failure_recovery(self, rag_orchestrator):
        """Test system behavior when services fail."""
        
        # Mock service failure
        with patch('services.embedding_service.EmbeddingService.generate_embeddings') as mock_embed:
            mock_embed.side_effect = Exception("Service unavailable")
            
            # Query should still work with fallback
            response = await rag_orchestrator.process_query(
                "Test query",
                enable_fallback=True
            )
            
            # Should use fallback mechanism
            assert response.success
            assert "fallback" in response.metadata.get("warnings", [])
    
    @pytest.mark.asyncio
    async def test_memory_limit_handling(self, rag_orchestrator):
        """Test handling of memory constraints."""
        
        # Create very large query
        large_query = "test " * 10000
        
        response = await rag_orchestrator.process_query(
            large_query,
            config={"max_query_length": 1000}
        )
        
        # Should truncate or reject gracefully
        assert response.success or "query too long" in response.error_message
```

## 📈 Coverage and Quality Metrics

### **Code Coverage Configuration**
```python
# .coveragerc
[run]
source = modern_rag_app
omit = 
    */tests/*
    */venv/*
    */migrations/*
    */__pycache__/*
    */config/local_settings.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    class .*\(Protocol\):
    @(abc\.)?abstractmethod

precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov
```

### **Quality Gates**
- **Code Coverage**: Minimum 80%
- **Performance**: 
  - Single query: < 500ms average
  - Concurrent queries: < 1000ms average
  - Memory usage: < 1GB per process
- **Reliability**: 
  - 99.9% test pass rate
  - Zero critical security vulnerabilities
  - Error rate < 0.1%

## 🚀 Running Tests

### **Basic Test Execution**
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m api                    # API tests only
pytest -m performance            # Performance tests only

# Run tests with coverage
pytest --cov=modern_rag_app --cov-report=html

# Run tests in parallel
pytest -n auto                   # Auto-detect CPU cores
pytest -n 4                      # Use 4 processes
```

### **Advanced Test Options**
```bash
# Run specific test file
pytest tests/test_services/test_query_analyzer.py

# Run specific test method
pytest tests/test_services/test_query_analyzer.py::TestQueryAnalyzer::test_simple_query

# Run with detailed output
pytest -v -s

# Run only failed tests from last run
pytest --lf

# Run tests modified since last commit
pytest --testmon

# Generate performance report
pytest -m performance --benchmark-only --benchmark-save=results
```

### **Continuous Integration**
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r tests/test_requirements.txt
    
    - name: Run tests
      run: |
        pytest --cov=modern_rag_app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 📚 Related Documentation

- **[Integration Testing Guide](integration.md)** - Detailed integration test setup
- **[API Testing](../api/testing.md)** - API-specific testing documentation
- **[Performance Testing](performance.md)** - Performance benchmark details
- **[Services Testing](../services/testing.md)** - Service layer test patterns

---

**The testing framework ensures comprehensive validation of the Modern RAG Application across all layers and use cases.**
