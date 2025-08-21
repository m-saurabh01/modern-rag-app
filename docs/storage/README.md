# Storage Documentation

## 📋 Overview

The Modern RAG Application uses a flexible storage architecture that supports multiple vector database backends and provides efficient file storage management for documents and processed data.

## 🗂️ Storage Components

### Vector Databases
- **[Vector Store Interface](vector_store.md)** - Abstract interface for vector database operations
- **ChromaDB Integration** - Default embedded vector database for development

### File Storage
- **Document Storage** - Raw uploaded files and metadata
- **Processed Storage** - Text extraction and analysis results  
- **Backup Storage** - Automated backup and recovery systems
- **Temporary Storage** - Processing intermediates and cache files

## 🏗️ Storage Architecture

```
Storage Layer
├── Vector Databases
│   ├── ChromaDB (Default)
│   ├── Qdrant (Optional)
│   └── Custom Implementations
├── File Storage
│   ├── Documents/
│   ├── Processed/
│   ├── Backups/
│   └── Temp/
└── Configuration
    ├── Environment Variables
    ├── Connection Settings
    └── Performance Tuning
```

## 🔧 Configuration

### Environment Variables
```bash
# Vector Database
VECTOR_DB_TYPE=chromadb
CHROMA_DB_PATH=./storage/chromadb
QDRANT_URL=localhost:6333

# File Storage
DOCUMENTS_PATH=./storage/documents
PROCESSED_PATH=./storage/processed
BACKUPS_PATH=./storage/backups
TEMP_PATH=./storage/temp

# Storage Limits
MAX_DOCUMENT_SIZE=50MB
MAX_STORAGE_SIZE=10GB
BACKUP_RETENTION_DAYS=30
```

## 📊 Storage Management

### File Organization
```
storage/
├── chromadb/           # Vector database files
├── documents/          # Original uploaded files
│   ├── pdf/
│   ├── images/
│   └── metadata/
├── processed/          # Processed content
│   ├── text/
│   ├── chunks/
│   └── embeddings/
├── backups/            # Automated backups
│   ├── daily/
│   ├── weekly/
│   └── monthly/
└── temp/               # Temporary processing files
```

### Storage Monitoring
- **Usage Tracking** - Monitor storage space and growth
- **Performance Metrics** - Database query performance
- **Health Checks** - Storage system availability
- **Cleanup Tasks** - Automated temp file cleanup

## 🔗 Related Documentation

- **[Vector Store Interface](vector_store.md)** - Technical implementation details
- **[Configuration Guide](../configuration.md)** - Environment setup
- **[System Architecture](../architecture.md)** - Overall system design

---

**The storage layer provides reliable, scalable data persistence for the Modern RAG Application.**
