# Chunking Service Documentation

## Overview

The `ChunkingService` provides modern semantic text chunking capabilities for the Modern RAG application. It replaces traditional fixed-size chunking with intelligent, content-aware strategies that preserve document structure and semantic coherence.

## Key Features

- **Multiple Chunking Strategies**: Recursive character, token-based, semantic, and hybrid approaches
- **Adaptive Chunking**: Automatically selects optimal strategy based on content analysis
- **Context Preservation**: Maintains document structure and relationships between chunks
- **Quality Validation**: Ensures chunk quality through overlap and coherence metrics
- **Metadata Enrichment**: Adds comprehensive metadata to each chunk for better retrieval

## Architecture

### Core Classes

#### ChunkingService
Main service class that orchestrates the chunking process.

**Key Methods:**
- `chunk_text()`: Main entry point for text chunking
- `chunk_documents()`: Batch processing for multiple documents
- `analyze_content()`: Content analysis for adaptive chunking
- `validate_chunks()`: Quality validation and scoring

#### ChunkingStrategy (Enum)
Defines available chunking strategies:
- `RECURSIVE_CHARACTER`: Hierarchical splitting respecting natural boundaries
- `CHARACTER`: Simple character-based splitting
- `TOKEN`: Token-aware splitting for language models
- `SEMANTIC`: Embedding-based semantic coherence splitting
- `HYBRID`: Combines multiple strategies for optimal results

#### TextChunk (Dataclass)
Represents a processed text chunk with metadata:
- `content`: The actual text content
- `metadata`: Document information, position, overlap details
- `chunk_id`: Unique identifier
- `start_char`/`end_char`: Position in original document

#### ChunkingConfig (Dataclass)
Configuration for chunking parameters:
- `strategy`: Selected chunking approach
- `chunk_size`: Target chunk size
- `chunk_overlap`: Overlap between chunks
- `adaptive_threshold`: Threshold for adaptive strategy selection

## Usage Examples

### Basic Chunking
```python
from services.chunking_service import ChunkingService, ChunkingStrategy

service = ChunkingService()
chunks = await service.chunk_text(
    text="Long document text...",
    strategy=ChunkingStrategy.RECURSIVE_CHARACTER
)
```

### Adaptive Chunking
```python
# Let the service choose the best strategy
chunks = await service.chunk_text(
    text="Document text...",
    adaptive=True
)
```

### Batch Processing
```python
documents = [
    {"content": "Doc 1 text...", "metadata": {"source": "doc1.pdf"}},
    {"content": "Doc 2 text...", "metadata": {"source": "doc2.pdf"}}
]

all_chunks = await service.chunk_documents(documents)
```

### Custom Configuration
```python
from services.chunking_service import ChunkingConfig

config = ChunkingConfig(
    strategy=ChunkingStrategy.SEMANTIC,
    chunk_size=512,
    chunk_overlap=50,
    preserve_formatting=True,
    min_chunk_size=100
)

chunks = await service.chunk_text(text, config=config)
```

## Configuration Options

### Strategy-Specific Settings

#### Recursive Character Chunking
- `separators`: List of separators in priority order
- `keep_separator`: Whether to keep separators in chunks
- `add_start_index`: Add character position tracking

#### Token-Based Chunking
- `encoding_name`: Tokenizer encoding (default: "cl100k_base")
- `model_name`: Specific model for tokenization
- `allowed_special`: Special tokens to preserve

#### Semantic Chunking
- `embedding_model`: Model for semantic similarity
- `similarity_threshold`: Threshold for semantic breaks
- `buffer_size`: Size of similarity calculation buffer

#### Hybrid Chunking
- `primary_strategy`: Main chunking approach
- `fallback_strategy`: Backup strategy for edge cases
- `quality_threshold`: Minimum quality score

### Quality Validation Settings
- `min_chunk_size`: Minimum acceptable chunk size
- `max_chunk_size`: Maximum chunk size before forced splitting
- `overlap_validation`: Validate chunk overlap consistency
- `coherence_threshold`: Minimum coherence score

## Performance Considerations

### Memory Usage
- Processes large documents in batches to manage memory
- Configurable batch size based on available RAM
- Automatic garbage collection for processed chunks

### Processing Speed
- Async processing for I/O operations
- Concurrent processing of multiple documents
- Caching of frequently used models and configurations

### Quality vs Speed Trade-offs
- **Fast Mode**: Character-based chunking for speed
- **Balanced Mode**: Recursive character with basic validation
- **High Quality Mode**: Semantic chunking with full validation

## Error Handling

The service uses comprehensive exception handling:

### ChunkingError
Base exception for chunking-related errors.
- `chunk_strategy`: Strategy that failed
- `text_length`: Length of text being processed

### Common Error Scenarios
1. **Text Too Short**: Input below minimum chunk size
2. **Strategy Failure**: Chunking strategy unable to process content
3. **Memory Limits**: Text too large for available memory
4. **Quality Validation**: Chunks fail quality thresholds

## Integration Points

### With Embedding Service
```python
# Process and embed in pipeline
chunks = await chunking_service.chunk_text(text)
embeddings = await embedding_service.embed_documents(chunks)
```

### With Vector Store
```python
# Direct integration for storage
chunks = await chunking_service.chunk_documents(documents)
await vector_store.add_documents(chunks)
```

### With Document Processor
```python
# Part of document processing pipeline
processed_docs = await document_processor.process_documents(files)
chunks = await chunking_service.chunk_documents(processed_docs)
```

## Testing

### Unit Tests
Located in `tests/test_services/test_chunking_service.py`

**Test Coverage:**
- Strategy selection and execution
- Adaptive chunking behavior
- Quality validation
- Error handling
- Performance benchmarks

### Integration Tests
- End-to-end chunking pipeline
- Integration with embedding service
- Vector store compatibility
- Memory usage validation

### Performance Tests
- Large document processing
- Memory usage profiling
- Strategy comparison benchmarks
- Concurrent processing validation

## Monitoring and Metrics

### Performance Metrics
- Chunks per second processing rate
- Memory usage per chunk
- Quality scores distribution
- Strategy selection frequency

### Health Checks
- Service initialization status
- Model loading verification
- Memory usage monitoring
- Error rate tracking

## Migration from Legacy System

### From Word-Based Chunking
```python
# Old approach
chunks = text.split()
chunks = [' '.join(chunks[i:i+chunk_size]) for i in range(0, len(chunks), step)]

# New approach
chunks = await chunking_service.chunk_text(
    text, 
    strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
    config=ChunkingConfig(chunk_size=chunk_size * 5)  # Approximate conversion
)
```

### Preserving Existing Metadata
The new service can preserve and enhance existing chunk metadata while improving quality.

## Best Practices

### Strategy Selection
1. **Academic Papers**: Use SEMANTIC for coherent concept preservation
2. **Technical Documentation**: Use RECURSIVE_CHARACTER for structure preservation
3. **Large Collections**: Use HYBRID for balanced quality and performance
4. **Real-time Processing**: Use CHARACTER for speed-critical applications

### Configuration Tuning
1. Start with default settings
2. Monitor quality metrics
3. Adjust based on retrieval performance
4. A/B test different strategies for your use case

### Performance Optimization
1. Use batch processing for multiple documents
2. Configure appropriate chunk sizes for your embedding model
3. Enable caching for frequently processed document types
4. Monitor memory usage and adjust batch sizes accordingly

## Troubleshooting

### Common Issues

1. **Poor Retrieval Quality**
   - Try semantic chunking
   - Increase chunk overlap
   - Validate chunk coherence scores

2. **Memory Errors**
   - Reduce batch size
   - Use character-based chunking for large documents
   - Enable streaming processing

3. **Slow Processing**
   - Switch to faster strategies
   - Reduce quality validation
   - Use parallel processing

4. **Inconsistent Chunks**
   - Enable quality validation
   - Use adaptive chunking
   - Review strategy configuration
