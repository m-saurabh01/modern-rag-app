# Modern RAG Application

A production-ready Retrieval-Augmented Generation (RAG) application built with FastAPI, featuring intelligent document processing, advanced query analysis, and multi-modal retrieval capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 32GB RAM recommended for optimal performance
- Local embedding models or API access

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd modern_rag_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration
```

### Configuration

Key configuration options in `.env`:

```bash
# Vector Database
VECTOR_DB_TYPE=chromadb  # Primary vector store
CHROMADB_PERSIST_DIRECTORY=./storage/chromadb

# Embedding Model (Local)
EMBEDDING_MODEL=/path/to/local/model/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

### Running the Application

```bash
# Start the API server
uvicorn api.main:app --reload --port 8000

# Access the application
# API Documentation: http://localhost:8000/docs
# Health Check: http://localhost:8000/health
```

## 📚 Complete System Documentation

For comprehensive developer documentation covering architecture, components, and workflows:

**📖 [Complete System Documentation](docs/README.md)**

The documentation includes:
- **System Architecture** - Complete component overview and data flow
- **Component Documentation** - Detailed service and module specifications  
- **API Reference** - Complete endpoint documentation
- **Developer Guides** - Setup, configuration, and customization
- **Processing Pipelines** - Document ingestion and query workflows

## 🔧 Core Features

### 🔍 Advanced Document Processing
- **Multi-format Support**: PDF extraction with OCR capabilities
- **Intelligent Chunking**: Content-aware semantic segmentation
- **Structure Preservation**: Maintains document hierarchy and relationships
- **Metadata Extraction**: Automatic content classification and tagging

### 🧠 Intelligent Query Processing  
- **Intent Classification**: Automatic query type detection and routing
- **Entity Recognition**: Extract domain-specific terms and entities
- **Query Enhancement**: Context expansion and semantic enrichment
- **Multi-language Support**: Configurable language processing

### 🎯 Advanced Retrieval System
- **Multi-Modal Search**: Search across text, tables, and structured data
- **Adaptive Strategies**: Query-specific retrieval optimization
- **Context-Aware Ranking**: Structure and semantic relevance scoring
- **Performance Modes**: Speed/Balanced/Accuracy configurations

### ⚡ Performance & Scalability
- **Streaming Processing**: Memory-efficient document handling
- **Multi-level Caching**: Results, embeddings, and pattern caching
- **Batch Operations**: Efficient bulk document processing
- **Resource Management**: Optimized for various hardware configurations

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │────│   FastAPI API   │────│   Services      │
│  (Frontend)     │    │   (Endpoints)   │    │  (Business)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Processing    │    │   Storage       │
                       │   Pipeline      │    │   Layer         │
                       └─────────────────┘    └─────────────────┘
```

### Layer Structure
```
├── api/           # FastAPI endpoints and request/response models
├── services/      # Core business logic and orchestration services
├── processing/    # Document processing and text analysis pipeline
├── storage/       # Vector database and file storage abstractions
├── models/        # Pydantic data models and validation schemas
├── config/        # Configuration management and structured logging
├── core/          # Core utilities, exceptions, and middleware
└── tests/         # Comprehensive test suite with integration tests
```

## 🧪 Testing

```bash
# Run complete test suite
pytest

# Run with coverage reporting
pytest --cov=. --cov-report=html

# Run specific components
pytest tests/test_services/     # Service layer tests
pytest tests/test_processing/   # Document processing tests
pytest tests/test_integration/  # End-to-end integration tests
```

## 🚀 Deployment

### Development Mode
```bash
uvicorn api.main:app --reload --port 8000
```

### Production Deployment
```bash
# With Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app

# With Docker
docker build -t modern-rag .
docker run -p 8000:8000 -v ./storage:/app/storage modern-rag
```

## 📊 Performance Characteristics

- **Query Response Time**: <500ms (Balanced mode), <300ms (Speed mode)
- **Document Processing**: 50+ documents/hour
- **Concurrent Users**: 100+ simultaneous queries supported
- **Memory Optimization**: Designed for 32GB RAM environments
- **Scalability**: 10GB → 700GB document corpus capability

## 🔒 Security Features

- **Input Validation**: Comprehensive Pydantic-based validation
- **Secure Processing**: Sandboxed document processing
- **CORS Configuration**: Configurable cross-origin policies
- **Error Handling**: Secure error responses without information leakage

## 🤝 Development

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with comprehensive tests
4. Run the test suite (`pytest`)
5. Submit a pull request with detailed description

### Code Standards
- **Type Hints**: Full type annotation required
- **Documentation**: Docstrings for all public methods
- **Testing**: Minimum 80% test coverage
- **Logging**: Structured logging throughout

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support & Documentation

- **📖 Complete Documentation**: [docs/README.md](docs/README.md)
- **🏗️ Architecture Guide**: [docs/architecture.md](docs/architecture.md)  
- **⚙️ Configuration Guide**: [docs/configuration.md](docs/configuration.md)
- **🔌 API Reference**: [docs/api/README.md](docs/api/README.md)

For issues, questions, and contributions:
1. Check the comprehensive documentation
2. Review existing GitHub issues
3. Create detailed issue reports with reproduction steps

---

**Built for production-scale intelligent document processing and retrieval** 🚀
