"""
Enhanced Chunking Service with Document Structure Awareness.

This module provides intelligent chunking strategies that leverage document structure
analysis for optimal content segmentation and context preservation.

Key Features:
- Structure-aware chunking using document analysis results
- Multiple chunking strategies (recursive, semantic, hybrid)
- Table-aware chunking that preserves tabular data integrity
- Section-based chunking that respects document hierarchy
- Adaptive chunk sizing based on content complexity
"""

import asyncio
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import re
from enum import Enum

# Simple text splitting functionality (replaces LangChain dependency)
def simple_recursive_split(
    text: str, 
    chunk_size: int, 
    chunk_overlap: int, 
    separators: List[str] = None
) -> List[str]:
    """
    Simple recursive text splitting without external dependencies.
    """
    if separators is None:
        separators = ['\n\n', '\n', '. ', '! ', '? ', ' ', '']
    
    def _split_text_with_separators(text: str, separators: List[str]) -> List[str]:
        """Split text using the first available separator."""
        if not separators or len(text) <= chunk_size:
            return [text] if text.strip() else []
        
        separator = separators[0]
        splits = text.split(separator)
        
        if len(splits) == 1:
            # Current separator didn't split the text, try next separator
            return _split_text_with_separators(text, separators[1:])
        
        # Merge splits into chunks
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Check if adding this split would exceed chunk size
            potential_chunk = current_chunk + separator + split if current_chunk else split
            
            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                # Current chunk is full, start a new one
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle case where single split is larger than chunk_size
                if len(split) > chunk_size:
                    # Recursively split the large split
                    sub_chunks = _split_text_with_separators(split, separators[1:])
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    return _split_text_with_separators(text, separators)


try:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        TokenTextSplitter
    )
    from langchain.schema import Document as LangChainDocument
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create simple placeholder classes
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str] = None, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ['\n\n', '\n', '. ', ' ', '']
        
        def split_text(self, text: str) -> List[str]:
            return simple_recursive_split(text, self.chunk_size, self.chunk_overlap, self.separators)
    
    class CharacterTextSplitter:
        def __init__(self, chunk_size: int, chunk_overlap: int, separator: str = '\n\n', **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separator = separator
        
        def split_text(self, text: str) -> List[str]:
            return simple_recursive_split(text, self.chunk_size, self.chunk_overlap, [self.separator])
    
    class TokenTextSplitter:
        def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        def split_text(self, text: str) -> List[str]:
            # Simple word-based splitting
            words = text.split()
            chunks = []
            current_chunk_words = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > self.chunk_size and current_chunk_words:
                    chunks.append(' '.join(current_chunk_words))
                    # Keep overlap
                    overlap_words = current_chunk_words[-self.chunk_overlap//10:] if self.chunk_overlap > 0 else []
                    current_chunk_words = overlap_words + [word]
                    current_length = sum(len(w) for w in current_chunk_words) + len(current_chunk_words)
                else:
                    current_chunk_words.append(word)
                    current_length += len(word) + 1
            
            if current_chunk_words:
                chunks.append(' '.join(current_chunk_words))
            
            return chunks
    
    class LangChainDocument:
        def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

from config import settings, get_logger, log_performance
from core.exceptions import ChunkingError, ValidationError
from .document_analyzer import (
    DocumentAnalyzer, DocumentStructure, DocumentSection, 
    TableStructure, SectionType
)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    RECURSIVE = "recursive"          # Default: respects document structure
    CHARACTER = "character"          # Simple character-based splitting
    TOKEN = "token"                  # Token-aware splitting
    SEMANTIC = "semantic"            # Content-aware semantic splitting
    STRUCTURE_AWARE = "structure_aware"  # Uses document structure analysis
    TABLE_AWARE = "table_aware"      # Preserves table integrity
    SECTION_BASED = "section_based"  # Chunks by document sections
    HYBRID = "hybrid"                # Combines multiple strategies


@dataclass
class ChunkingConfig:
    """Configuration for chunking operations."""
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 1000
    chunk_overlap: int = 200
    length_function: Callable = len
    separators: Optional[List[str]] = None
    keep_separator: bool = True
    add_start_index: bool = True
    strip_whitespace: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValidationError(
                "Chunk overlap must be less than chunk size",
                field_name="chunk_overlap",
                field_value=self.chunk_overlap
            )
        
        if self.chunk_size < 50:
            raise ValidationError(
                "Chunk size must be at least 50 characters",
                field_name="chunk_size", 
                field_value=self.chunk_size
            )


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    content: str
    chunk_id: str
    source_document: str
    start_index: int = 0
    end_index: int = 0
    chunk_number: int = 0
    total_chunks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    prev_context: Optional[str] = None
    next_context: Optional[str] = None
    
    def __post_init__(self):
        """Calculate end_index if not provided."""
        if self.end_index == 0:
            self.end_index = self.start_index + len(self.content)
    
    @property
    def length(self) -> int:
        """Get chunk length."""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'source_document': self.source_document,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'chunk_number': self.chunk_number,
            'total_chunks': self.total_chunks,
            'length': self.length,
            'word_count': self.word_count,
            'metadata': self.metadata,
            'prev_context': self.prev_context,
            'next_context': self.next_context,
        }


class ChunkingService:
    """
    Modern text chunking service with semantic awareness.
    
    Features:
    - Multiple chunking strategies (recursive, semantic, token-based)
    - Content-aware splitting that respects document structure
    - Context preservation with overlapping chunks
    - Metadata enrichment with document structure information
    - Progress tracking for large documents
    - Memory-efficient processing for large texts
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize chunking service.
        
        Args:
            config: Chunking configuration. Uses defaults if None.
        """
        self.config = config or ChunkingConfig(
            chunk_size=settings.processing.chunk_size,
            chunk_overlap=settings.processing.chunk_overlap
        )
        
        self.logger = get_logger(__name__)
        
        # Initialize document analyzer for structure-aware chunking
        self.document_analyzer = DocumentAnalyzer()
        
        # Performance tracking
        self._stats = {
            'total_documents_processed': 0,
            'total_chunks_created': 0,
            'average_chunks_per_document': 0.0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
        
        self.logger.info(
            "ChunkingService initialized",
            strategy=self.config.strategy.value,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()
    
    def _create_text_splitter(self, strategy: ChunkingStrategy) -> Union[
        RecursiveCharacterTextSplitter, 
        CharacterTextSplitter, 
        TokenTextSplitter
    ]:
        """
        Create appropriate text splitter based on strategy.
        
        Args:
            strategy: Chunking strategy to use
            
        Returns:
            Configured text splitter instance
            
        Raises:
            ChunkingError: If unsupported strategy specified
        """
        base_kwargs = {
            'chunk_size': self.config.chunk_size,
            'chunk_overlap': self.config.chunk_overlap,
            'length_function': self.config.length_function,
            'add_start_index': self.config.add_start_index,
        }
        
        if strategy == ChunkingStrategy.RECURSIVE:
            # Default separators for document structure
            separators = self.config.separators or [
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ". ",        # Sentence endings
                "! ",        # Exclamations
                "? ",        # Questions
                ";",         # Semicolons
                ",",         # Commas
                " ",         # Spaces
                ""           # Characters
            ]
            
            return RecursiveCharacterTextSplitter(
                separators=separators,
                keep_separator=self.config.keep_separator,
                strip_whitespace=self.config.strip_whitespace,
                **base_kwargs
            )
        
        elif strategy == ChunkingStrategy.CHARACTER:
            separator = self.config.separators[0] if self.config.separators else "\n\n"
            return CharacterTextSplitter(
                separator=separator,
                strip_whitespace=self.config.strip_whitespace,
                **base_kwargs
            )
        
        elif strategy == ChunkingStrategy.TOKEN:
            return TokenTextSplitter(
                **base_kwargs
            )
        
        else:
            raise ChunkingError(
                f"Unsupported chunking strategy: {strategy}",
                chunk_strategy=strategy.value
            )
    
    def _generate_chunk_id(
        self, 
        document_name: str, 
        chunk_number: int,
        strategy: str = None
    ) -> str:
        """
        Generate unique chunk identifier.
        
        Args:
            document_name: Source document name
            chunk_number: Chunk sequence number
            strategy: Chunking strategy used (optional)
            
        Returns:
            Unique chunk identifier
        """
        # Clean document name for ID
        clean_name = re.sub(r'[^\w\-_.]', '_', document_name)
        
        if strategy:
            return f"{clean_name}_{strategy}_{chunk_number:04d}"
        else:
            return f"{clean_name}_{chunk_number:04d}"
    
    def _add_context_information(
        self, 
        chunks: List[TextChunk],
        context_size: int = 50
    ) -> List[TextChunk]:
        """
        Add previous and next context to chunks.
        
        Args:
            chunks: List of text chunks
            context_size: Number of characters for context
            
        Returns:
            Chunks with added context information
        """
        for i, chunk in enumerate(chunks):
            # Previous context
            if i > 0:
                prev_chunk = chunks[i - 1]
                chunk.prev_context = prev_chunk.content[-context_size:] if len(prev_chunk.content) > context_size else prev_chunk.content
            
            # Next context
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                chunk.next_context = next_chunk.content[:context_size] if len(next_chunk.content) > context_size else next_chunk.content
        
        return chunks
    
    def _extract_document_structure(self, text: str) -> Dict[str, Any]:
        """
        Extract document structure information.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with structure information
        """
        structure = {
            'total_length': len(text),
            'word_count': len(text.split()),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'line_count': len(text.split('\n')),
            'has_headers': bool(re.search(r'^#+\s', text, re.MULTILINE)),
            'has_lists': bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE)),
            'has_numbers': bool(re.search(r'^\s*\d+\.', text, re.MULTILINE)),
        }
        
        # Extract potential headers
        headers = re.findall(r'^(#{1,6}\s+.+)$', text, re.MULTILINE)
        structure['headers'] = headers[:10]  # Limit to first 10 headers
        
        return structure
    
    @log_performance("document_chunking")
    async def chunk_text(
        self,
        text: str,
        document_name: str,
        strategy: Optional[ChunkingStrategy] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TextChunk]:
        """
        Chunk text using specified strategy.
        
        Args:
            text: Text to chunk
            document_name: Source document identifier
            strategy: Chunking strategy (uses config default if None)
            progress_callback: Optional progress callback (current, total)
            
        Returns:
            List of text chunks with metadata
            
        Raises:
            ChunkingError: If chunking fails
            ValidationError: If text is empty or invalid
        """
        if not text or not text.strip():
            raise ValidationError(
                "Text cannot be empty",
                field_name="text",
                field_value=""
            )
        
        strategy = strategy or self.config.strategy
        start_time = time.time()
        
        try:
            # Extract document structure
            doc_structure = self._extract_document_structure(text)
            
            # Create appropriate text splitter
            splitter = self._create_text_splitter(strategy)
            
            # Create LangChain document
            doc = LangChainDocument(
                page_content=text,
                metadata={'source': document_name}
            )
            
            # Split document
            split_docs = await asyncio.get_event_loop().run_in_executor(
                None, splitter.split_documents, [doc]
            )
            
            # Convert to TextChunk objects
            chunks = []
            total_chunks = len(split_docs)
            
            for i, split_doc in enumerate(split_docs):
                # Progress callback
                if progress_callback:
                    progress_callback(i, total_chunks)
                
                chunk = TextChunk(
                    content=split_doc.page_content,
                    chunk_id=self._generate_chunk_id(document_name, i + 1, strategy.value),
                    source_document=document_name,
                    start_index=split_doc.metadata.get('start_index', 0),
                    chunk_number=i + 1,
                    total_chunks=total_chunks,
                    metadata={
                        'strategy': strategy.value,
                        'document_structure': doc_structure,
                        'creation_time': time.time()
                    }
                )
                chunks.append(chunk)
            
            # Add context information
            chunks = self._add_context_information(chunks)
            
            # Final progress callback
            if progress_callback:
                progress_callback(total_chunks, total_chunks)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._stats['total_documents_processed'] += 1
            self._stats['total_chunks_created'] += len(chunks)
            self._stats['total_processing_time'] += processing_time
            
            if self._stats['total_documents_processed'] > 0:
                self._stats['average_chunks_per_document'] = (
                    self._stats['total_chunks_created'] / self._stats['total_documents_processed']
                )
                self._stats['average_processing_time'] = (
                    self._stats['total_processing_time'] / self._stats['total_documents_processed']
                )
            
            self.logger.info(
                "Document chunked successfully",
                document_name=document_name,
                strategy=strategy.value,
                total_chunks=len(chunks),
                processing_time_ms=round(processing_time * 1000, 2),
                avg_chunk_size=sum(chunk.length for chunk in chunks) // len(chunks) if chunks else 0
            )
            
            return chunks
            
        except Exception as e:
            if isinstance(e, (ChunkingError, ValidationError)):
                raise
            else:
                raise ChunkingError(
                    f"Failed to chunk document '{document_name}': {str(e)}",
                    chunk_strategy=strategy.value,
                    text_length=len(text),
                    cause=e
                )
    
    async def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = 'content',
        document_id_field: str = 'id',
        strategy: Optional[ChunkingStrategy] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TextChunk]:
        """
        Chunk multiple documents with batch processing.
        
        Args:
            documents: List of document dictionaries
            text_field: Field containing text content
            document_id_field: Field containing document identifier
            strategy: Chunking strategy to use
            progress_callback: Optional progress callback
            
        Returns:
            List of all chunks from all documents
            
        Raises:
            ChunkingError: If chunking fails
        """
        if not documents:
            return []
        
        all_chunks = []
        total_docs = len(documents)
        
        self.logger.info(
            "Starting batch document chunking",
            total_documents=total_docs,
            strategy=strategy.value if strategy else self.config.strategy.value
        )
        
        for i, doc in enumerate(documents):
            # Progress callback for overall progress
            if progress_callback:
                progress_callback(i, total_docs)
            
            # Skip documents without required fields
            if text_field not in doc or document_id_field not in doc:
                self.logger.warning(
                    "Skipping document missing required fields",
                    required_fields=[text_field, document_id_field],
                    available_fields=list(doc.keys())
                )
                continue
            
            text = doc[text_field]
            doc_id = doc[document_id_field]
            
            if not text or not str(text).strip():
                self.logger.warning(
                    "Skipping document with empty text",
                    document_id=doc_id
                )
                continue
            
            try:
                # Create per-document progress callback
                def doc_progress(current, total):
                    # Could emit more granular progress here if needed
                    pass
                
                # Chunk individual document
                doc_chunks = await self.chunk_text(
                    text=str(text),
                    document_name=str(doc_id),
                    strategy=strategy,
                    progress_callback=doc_progress
                )
                
                # Add document metadata to each chunk
                for chunk in doc_chunks:
                    chunk.metadata.update({
                        'original_document': doc,
                        'document_id': doc_id,
                        'batch_processing': True
                    })
                
                all_chunks.extend(doc_chunks)
                
            except Exception as e:
                self.logger.error(
                    "Failed to chunk document",
                    document_id=doc_id,
                    error=str(e),
                    error_type=type(e).__name__
                )
                # Continue with other documents rather than failing entirely
                continue
        
        # Final progress callback
        if progress_callback:
            progress_callback(total_docs, total_docs)
        
        self.logger.info(
            "Batch document chunking completed",
            total_documents=total_docs,
            total_chunks=len(all_chunks),
            average_chunks_per_doc=len(all_chunks) / total_docs if total_docs > 0 else 0
        )
        
        return all_chunks
    
    async def chunk_with_adaptive_strategy(
        self,
        text: str,
        document_name: str,
        target_chunk_count: Optional[int] = None,
        max_chunk_size: Optional[int] = None
    ) -> List[TextChunk]:
        """
        Chunk text with adaptive strategy based on content characteristics.
        
        Args:
            text: Text to chunk
            document_name: Source document identifier
            target_chunk_count: Desired number of chunks (approximate)
            max_chunk_size: Maximum chunk size (overrides config)
            
        Returns:
            List of optimally chunked text
        """
        # Analyze text characteristics
        structure = self._extract_document_structure(text)
        
        # Determine optimal strategy based on content
        if structure['has_headers'] and structure['paragraph_count'] > 10:
            # Structured document - use recursive splitting
            strategy = ChunkingStrategy.RECURSIVE
        elif structure['line_count'] / structure['paragraph_count'] > 5:
            # Dense text - use character splitting
            strategy = ChunkingStrategy.CHARACTER
        else:
            # Default to recursive
            strategy = ChunkingStrategy.RECURSIVE
        
        # Adjust chunk size if target count specified
        original_chunk_size = self.config.chunk_size
        if target_chunk_count:
            estimated_chunk_size = structure['total_length'] / target_chunk_count
            # Ensure chunk size is within reasonable bounds
            adaptive_chunk_size = max(200, min(2000, int(estimated_chunk_size)))
            self.config.chunk_size = adaptive_chunk_size
        
        if max_chunk_size:
            self.config.chunk_size = min(self.config.chunk_size, max_chunk_size)
        
        try:
            chunks = await self.chunk_text(
                text=text,
                document_name=document_name,
                strategy=strategy
            )
            
            self.logger.info(
                "Adaptive chunking completed",
                document_name=document_name,
                strategy=strategy.value,
                original_chunk_size=original_chunk_size,
                adaptive_chunk_size=self.config.chunk_size,
                target_chunks=target_chunk_count,
                actual_chunks=len(chunks)
            )
            
            return chunks
            
        finally:
            # Restore original chunk size
            self.config.chunk_size = original_chunk_size
    
    def validate_chunks(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """
        Validate chunk quality and consistency.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Validation report with quality metrics
        """
        if not chunks:
            return {'valid': False, 'error': 'No chunks provided'}
        
        report = {
            'valid': True,
            'total_chunks': len(chunks),
            'issues': [],
            'metrics': {}
        }
        
        # Check for empty chunks
        empty_chunks = [chunk for chunk in chunks if not chunk.content.strip()]
        if empty_chunks:
            report['issues'].append(f"{len(empty_chunks)} empty chunks found")
        
        # Check chunk size distribution
        chunk_sizes = [chunk.length for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        
        report['metrics'].update({
            'average_chunk_size': avg_size,
            'min_chunk_size': min_size,
            'max_chunk_size': max_size,
            'size_std_dev': (sum((s - avg_size) ** 2 for s in chunk_sizes) / len(chunk_sizes)) ** 0.5
        })
        
        # Check for very small chunks (potential splitting issues)
        small_chunks = [chunk for chunk in chunks if chunk.length < self.config.chunk_size * 0.1]
        if small_chunks:
            report['issues'].append(f"{len(small_chunks)} unusually small chunks found")
        
        # Check for very large chunks (splitting may have failed)
        large_chunks = [chunk for chunk in chunks if chunk.length > self.config.chunk_size * 1.5]
        if large_chunks:
            report['issues'].append(f"{len(large_chunks)} oversized chunks found")
        
        # Check chunk continuity (start/end indices)
        for i in range(1, len(chunks)):
            if chunks[i - 1].source_document == chunks[i].source_document:
                if chunks[i - 1].end_index > chunks[i].start_index:
                    # This is expected for overlapping chunks
                    pass
                elif chunks[i - 1].end_index < chunks[i].start_index - 100:
                    # Large gap might indicate missing content
                    report['issues'].append(f"Large gap between chunks {i-1} and {i}")
        
        # Overall quality assessment
        if len(report['issues']) > len(chunks) * 0.1:  # More than 10% issues
            report['valid'] = False
            report['error'] = "Too many quality issues detected"
        
        return report


    # ==================== ENHANCED CHUNKING METHODS (PHASE 3.3b) ====================
    
    @log_performance("structure_aware_chunking")
    async def chunk_with_structure_analysis(
        self,
        text: str,
        document_id: str,
        strategy: Optional[ChunkingStrategy] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[TextChunk], DocumentStructure]:
        """
        Chunk text using document structure analysis for optimal segmentation.
        
        Args:
            text: Text content to chunk
            document_id: Unique document identifier
            strategy: Chunking strategy (defaults to STRUCTURE_AWARE)
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (chunks, document_structure_analysis)
        """
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty", field_name="text", field_value="")
        
        start_time = time.time()
        strategy = strategy or ChunkingStrategy.STRUCTURE_AWARE
        
        self.logger.info(
            "Starting structure-aware chunking",
            extra={
                'document_id': document_id,
                'text_length': len(text),
                'strategy': strategy.value
            }
        )
        
        try:
            # Step 1: Analyze document structure
            if progress_callback:
                progress_callback(1, 6)
            
            doc_structure = self.document_analyzer.analyze_document(text, document_id)
            
            self.logger.debug(
                "Document structure analysis complete",
                extra={
                    'sections': len(doc_structure.sections),
                    'tables': len(doc_structure.tables),
                    'entities': len(doc_structure.entities),
                    'document_type': doc_structure.document_type
                }
            )
            
            # Step 2: Choose appropriate chunking based on structure
            if progress_callback:
                progress_callback(2, 6)
                
            if strategy == ChunkingStrategy.STRUCTURE_AWARE:
                chunks = await self._chunk_by_structure(text, doc_structure, progress_callback)
            elif strategy == ChunkingStrategy.TABLE_AWARE:
                chunks = await self._chunk_table_aware(text, doc_structure, progress_callback)
            elif strategy == ChunkingStrategy.SECTION_BASED:
                chunks = await self._chunk_by_sections(text, doc_structure, progress_callback)
            else:
                # Fallback to regular chunking with structure metadata
                chunks = await self.chunk_text(text, document_id, strategy, progress_callback)
                # Add structure metadata to chunks
                chunks = self._enrich_chunks_with_structure(chunks, doc_structure)
            
            if progress_callback:
                progress_callback(6, 6)
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                "Structure-aware chunking completed",
                extra={
                    'document_id': document_id,
                    'chunks_created': len(chunks),
                    'processing_time': f"{processing_time:.3f}s",
                    'document_type': doc_structure.document_type,
                    'complexity_score': doc_structure.metadata.get('complexity_score', 0)
                }
            )
            
            return chunks, doc_structure
            
        except Exception as e:
            self.logger.error(
                "Structure-aware chunking failed",
                extra={
                    'document_id': document_id,
                    'error': str(e),
                    'text_length': len(text)
                }
            )
            raise ChunkingError(
                f"Structure-aware chunking failed for document {document_id}: {str(e)}",
                chunk_strategy=strategy.value,
                text_length=len(text),
                cause=e
            )
    
    async def _chunk_by_structure(
        self, 
        text: str, 
        doc_structure: DocumentStructure,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TextChunk]:
        """
        Chunk text based on document structure analysis.
        Preserves section boundaries and table integrity.
        """
        chunks = []
        current_step = 3
        total_steps = 6
        
        # Sort sections by position for sequential processing
        sections = sorted(doc_structure.sections, key=lambda s: s.start_position)
        
        for i, section in enumerate(sections):
            if progress_callback and i % 10 == 0:  # Update progress periodically
                progress = current_step + (i / len(sections))
                progress_callback(int(progress), total_steps)
            
            section_text = text[section.start_position:section.end_position]
            
            # Handle different section types differently
            if section.section_type == SectionType.TABLE:
                # Tables get special handling
                table_chunks = await self._chunk_table_content(
                    section_text, section, doc_structure.tables
                )
                chunks.extend(table_chunks)
            
            elif section.section_type in [SectionType.HEADER, SectionType.SUBHEADER]:
                # Headers typically stay with following content
                header_chunk = await self._create_header_chunk(
                    section, sections, text, doc_structure
                )
                if header_chunk:
                    chunks.append(header_chunk)
            
            elif section.section_type == SectionType.LIST_ITEM:
                # Group related list items together
                list_chunks = await self._chunk_list_items(
                    section, sections, text, doc_structure
                )
                chunks.extend(list_chunks)
            
            else:
                # Regular paragraphs - chunk if too large
                if len(section_text) > self.config.chunk_size:
                    para_chunks = await self._chunk_large_section(
                        section_text, section, doc_structure
                    )
                    chunks.extend(para_chunks)
                else:
                    chunk = self._create_section_chunk(section_text, section, doc_structure)
                    chunks.append(chunk)
        
        # Post-process: merge small adjacent chunks
        chunks = await self._merge_small_chunks(chunks)
        
        return chunks
    
    async def _chunk_table_aware(
        self,
        text: str,
        doc_structure: DocumentStructure,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TextChunk]:
        """
        Chunk text while preserving table integrity and structure.
        """
        chunks = []
        processed_positions = set()
        
        # Process tables first to preserve their integrity
        for table in doc_structure.tables:
            table_text = text[table.position:table.position + len(str(table.headers + table.rows))]
            
            # Create table chunk with full context
            table_chunk = TextChunk(
                content=table_text,
                chunk_id=f"{doc_structure.document_id}_table_{table.table_id}",
                metadata={
                    'chunk_type': 'table',
                    'table_id': table.table_id,
                    'table_type': table.table_type,
                    'column_count': table.column_count,
                    'row_count': table.row_count,
                    'headers': table.headers,
                    'document_type': doc_structure.document_type,
                    'is_complete_table': True
                },
                start_position=table.position,
                end_position=table.position + len(table_text),
                document_name=doc_structure.document_id,
                length=len(table_text)
            )
            chunks.append(table_chunk)
            
            # Mark table positions as processed
            for pos in range(table.position, table.position + len(table_text)):
                processed_positions.add(pos)
        
        # Process non-table content
        sections = sorted(doc_structure.sections, key=lambda s: s.start_position)
        
        for section in sections:
            # Skip if this section overlaps with a table
            if any(pos in processed_positions 
                   for pos in range(section.start_position, section.end_position)):
                continue
            
            section_text = text[section.start_position:section.end_position]
            
            if len(section_text) > self.config.chunk_size:
                # Chunk large sections while being table-aware
                section_chunks = await self._chunk_large_section(
                    section_text, section, doc_structure
                )
                chunks.extend(section_chunks)
            else:
                chunk = self._create_section_chunk(section_text, section, doc_structure)
                chunks.append(chunk)
        
        return sorted(chunks, key=lambda c: c.start_position)
    
    async def _chunk_by_sections(
        self,
        text: str,
        doc_structure: DocumentStructure,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TextChunk]:
        """
        Chunk text by document sections, respecting hierarchical structure.
        """
        chunks = []
        
        # Group sections by hierarchy level
        sections_by_level = {}
        for section in doc_structure.sections:
            level = section.level
            if level not in sections_by_level:
                sections_by_level[level] = []
            sections_by_level[level].append(section)
        
        # Process sections level by level
        for level in sorted(sections_by_level.keys()):
            level_sections = sorted(sections_by_level[level], key=lambda s: s.start_position)
            
            for section in level_sections:
                # Include child sections in the chunk
                section_content = await self._get_section_with_children(
                    section, doc_structure.sections, text
                )
                
                if len(section_content) > self.config.chunk_size * 1.5:
                    # Large section: create multiple chunks but preserve context
                    section_chunks = await self._chunk_large_hierarchical_section(
                        section_content, section, doc_structure
                    )
                    chunks.extend(section_chunks)
                else:
                    # Small section: single chunk
                    chunk = TextChunk(
                        content=section_content,
                        chunk_id=f"{doc_structure.document_id}_section_{section.section_id}",
                        metadata={
                            'chunk_type': 'section',
                            'section_id': section.section_id,
                            'section_type': section.section_type.value,
                            'section_level': section.level,
                            'section_title': section.title,
                            'has_children': len(section.children_ids) > 0,
                            'parent_id': section.parent_id,
                            'document_type': doc_structure.document_type
                        },
                        start_position=section.start_position,
                        end_position=section.end_position,
                        document_name=doc_structure.document_id,
                        length=len(section_content)
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _enrich_chunks_with_structure(
        self,
        chunks: List[TextChunk],
        doc_structure: DocumentStructure
    ) -> List[TextChunk]:
        """
        Enrich existing chunks with document structure information.
        """
        for chunk in chunks:
            # Find which section(s) this chunk overlaps with
            overlapping_sections = []
            overlapping_tables = []
            overlapping_entities = []
            
            for section in doc_structure.sections:
                if (chunk.start_position <= section.end_position and 
                    chunk.end_position >= section.start_position):
                    overlapping_sections.append({
                        'section_id': section.section_id,
                        'section_type': section.section_type.value,
                        'level': section.level,
                        'title': section.title
                    })
            
            for table in doc_structure.tables:
                if (chunk.start_position <= table.position + 1000 and  # Approximate table size
                    chunk.end_position >= table.position):
                    overlapping_tables.append({
                        'table_id': table.table_id,
                        'table_type': table.table_type,
                        'columns': len(table.headers)
                    })
            
            for entity in doc_structure.entities:
                if (chunk.start_position <= entity.position and 
                    chunk.end_position >= entity.position):
                    overlapping_entities.append({
                        'entity_type': entity.entity_type.value,
                        'text': entity.text,
                        'confidence': entity.confidence
                    })
            
            # Add structure information to metadata
            chunk.metadata.update({
                'document_structure': {
                    'document_type': doc_structure.document_type,
                    'overlapping_sections': overlapping_sections,
                    'overlapping_tables': overlapping_tables,
                    'overlapping_entities': overlapping_entities,
                    'complexity_score': doc_structure.metadata.get('complexity_score', 0)
                }
            })
        
        return chunks
    
    # Helper methods for structure-aware chunking
    
    async def _chunk_table_content(
        self,
        table_text: str,
        section: DocumentSection,
        tables: List[TableStructure]
    ) -> List[TextChunk]:
        """Create optimized chunks for table content."""
        # Find the specific table
        table = None
        for t in tables:
            if abs(t.position - section.start_position) < 100:  # Close position match
                table = t
                break
        
        if not table:
            # Fallback: treat as regular content
            return [self._create_section_chunk(table_text, section, None)]
        
        chunks = []
        
        # Strategy 1: If table is small, keep as single chunk
        if table.row_count <= 10 and len(table_text) <= self.config.chunk_size:
            chunk = TextChunk(
                content=table_text,
                chunk_id=f"table_{table.table_id}_complete",
                metadata={
                    'chunk_type': 'complete_table',
                    'table_id': table.table_id,
                    'table_type': table.table_type,
                    'column_count': table.column_count,
                    'row_count': table.row_count
                },
                start_position=section.start_position,
                end_position=section.end_position,
                document_name=section.section_id.split('_')[0],
                length=len(table_text)
            )
            chunks.append(chunk)
        
        # Strategy 2: Large table - split by rows but keep headers
        else:
            header_text = " | ".join(table.headers)
            rows_per_chunk = max(1, (self.config.chunk_size - len(header_text)) // 100)
            
            for i in range(0, len(table.rows), rows_per_chunk):
                chunk_rows = table.rows[i:i + rows_per_chunk]
                chunk_content = header_text + "\n" + "\n".join([" | ".join(row) for row in chunk_rows])
                
                chunk = TextChunk(
                    content=chunk_content,
                    chunk_id=f"table_{table.table_id}_part_{i // rows_per_chunk + 1}",
                    metadata={
                        'chunk_type': 'table_segment',
                        'table_id': table.table_id,
                        'table_type': table.table_type,
                        'segment_number': i // rows_per_chunk + 1,
                        'total_segments': (len(table.rows) + rows_per_chunk - 1) // rows_per_chunk,
                        'has_complete_headers': True
                    },
                    start_position=section.start_position + i * 50,  # Approximate position
                    end_position=section.start_position + (i + len(chunk_rows)) * 50,
                    document_name=section.section_id.split('_')[0],
                    length=len(chunk_content)
                )
                chunks.append(chunk)
        
        return chunks
    
    async def _create_header_chunk(
        self,
        header_section: DocumentSection,
        all_sections: List[DocumentSection],
        text: str,
        doc_structure: DocumentStructure
    ) -> Optional[TextChunk]:
        """Create a chunk for a header that includes following content."""
        # Find content that belongs with this header
        header_end = header_section.end_position
        next_header_start = len(text)  # Default to end of document
        
        # Find next header of same or higher level
        for section in all_sections:
            if (section.start_position > header_end and 
                section.section_type in [SectionType.HEADER, SectionType.SUBHEADER] and
                section.level <= header_section.level):
                next_header_start = section.start_position
                break
        
        # Extract content including the header and following content
        content_length = min(self.config.chunk_size, next_header_start - header_section.start_position)
        chunk_content = text[header_section.start_position:header_section.start_position + content_length]
        
        if len(chunk_content.strip()) < 10:  # Too small
            return None
        
        return TextChunk(
            content=chunk_content,
            chunk_id=f"header_{header_section.section_id}",
            metadata={
                'chunk_type': 'header_with_content',
                'section_id': header_section.section_id,
                'section_type': header_section.section_type.value,
                'level': header_section.level,
                'title': header_section.title,
                'includes_following_content': True
            },
            start_position=header_section.start_position,
            end_position=header_section.start_position + len(chunk_content),
            document_name=doc_structure.document_id,
            length=len(chunk_content)
        )
    
    async def _chunk_list_items(
        self,
        list_section: DocumentSection,
        all_sections: List[DocumentSection],
        text: str,
        doc_structure: DocumentStructure
    ) -> List[TextChunk]:
        """Group related list items into coherent chunks."""
        # Find all consecutive list items
        list_sections = [list_section]
        
        # Look for adjacent list items
        for section in all_sections:
            if (section.section_type == SectionType.LIST_ITEM and 
                section.start_position > list_section.end_position and
                section.start_position - list_section.end_position < 100):  # Close proximity
                list_sections.append(section)
        
        # Group list items into chunks
        chunks = []
        current_chunk_items = []
        current_chunk_length = 0
        
        for item_section in list_sections:
            item_text = text[item_section.start_position:item_section.end_position]
            
            if current_chunk_length + len(item_text) > self.config.chunk_size and current_chunk_items:
                # Create chunk from current items
                chunk_content = "\n".join(current_chunk_items)
                chunks.append(self._create_list_chunk(chunk_content, len(chunks), doc_structure))
                
                # Start new chunk
                current_chunk_items = [item_text]
                current_chunk_length = len(item_text)
            else:
                current_chunk_items.append(item_text)
                current_chunk_length += len(item_text)
        
        # Create final chunk if there are remaining items
        if current_chunk_items:
            chunk_content = "\n".join(current_chunk_items)
            chunks.append(self._create_list_chunk(chunk_content, len(chunks), doc_structure))
        
        return chunks
    
    def _create_list_chunk(self, content: str, chunk_index: int, doc_structure: DocumentStructure) -> TextChunk:
        """Create a chunk for list content."""
        return TextChunk(
            content=content,
            chunk_id=f"list_{doc_structure.document_id}_{chunk_index}",
            metadata={
                'chunk_type': 'list_items',
                'item_count': len([line for line in content.split('\n') if line.strip()]),
                'document_type': doc_structure.document_type
            },
            start_position=0,  # Will be updated if needed
            end_position=len(content),
            document_name=doc_structure.document_id,
            length=len(content)
        )
    
    async def _chunk_large_section(
        self,
        section_text: str,
        section: DocumentSection,
        doc_structure: DocumentStructure
    ) -> List[TextChunk]:
        """Chunk a large section while preserving context."""
        # Use recursive text splitter for large sections
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=['\n\n', '\n', '. ', '! ', '? ', ' ', '']
        )
        
        text_chunks = splitter.split_text(section_text)
        chunks = []
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = TextChunk(
                content=chunk_text,
                chunk_id=f"section_{section.section_id}_part_{i+1}",
                metadata={
                    'chunk_type': 'large_section_part',
                    'section_id': section.section_id,
                    'section_type': section.section_type.value,
                    'part_number': i + 1,
                    'total_parts': len(text_chunks),
                    'original_section_title': section.title
                },
                start_position=section.start_position + i * len(chunk_text),  # Approximate
                end_position=section.start_position + (i + 1) * len(chunk_text),
                document_name=doc_structure.document_id,
                length=len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_section_chunk(
        self,
        section_text: str,
        section: DocumentSection,
        doc_structure: Optional[DocumentStructure]
    ) -> TextChunk:
        """Create a chunk for a single section."""
        return TextChunk(
            content=section_text,
            chunk_id=f"section_{section.section_id}",
            metadata={
                'chunk_type': 'section',
                'section_id': section.section_id,
                'section_type': section.section_type.value,
                'level': section.level,
                'title': section.title,
                'document_type': doc_structure.document_type if doc_structure else 'unknown'
            },
            start_position=section.start_position,
            end_position=section.end_position,
            document_name=doc_structure.document_id if doc_structure else 'unknown',
            length=len(section_text)
        )
    
    async def _merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Merge chunks that are too small for effective processing."""
        if not chunks:
            return chunks
        
        min_chunk_size = self.config.chunk_size // 4  # 25% of target size
        merged_chunks = []
        current_merged = None
        
        for chunk in sorted(chunks, key=lambda c: c.start_position):
            if chunk.length < min_chunk_size:
                if current_merged is None:
                    current_merged = chunk
                else:
                    # Merge with previous small chunk
                    merged_content = current_merged.content + "\n\n" + chunk.content
                    if len(merged_content) <= self.config.chunk_size * 1.2:  # Allow 20% overflow
                        current_merged = TextChunk(
                            content=merged_content,
                            chunk_id=f"merged_{current_merged.chunk_id}_{chunk.chunk_id}",
                            metadata={
                                **current_merged.metadata,
                                'merged_from': [current_merged.chunk_id, chunk.chunk_id],
                                'chunk_type': 'merged_small_chunks'
                            },
                            start_position=current_merged.start_position,
                            end_position=chunk.end_position,
                            document_name=current_merged.document_name,
                            length=len(merged_content)
                        )
                    else:
                        # Can't merge - add current and start new
                        merged_chunks.append(current_merged)
                        current_merged = chunk
            else:
                # Add any pending merged chunk
                if current_merged is not None:
                    merged_chunks.append(current_merged)
                    current_merged = None
                
                # Add the current normal-sized chunk
                merged_chunks.append(chunk)
        
        # Add final merged chunk if exists
        if current_merged is not None:
            merged_chunks.append(current_merged)
        
        return merged_chunks
    
    async def _get_section_with_children(
        self,
        section: DocumentSection,
        all_sections: List[DocumentSection],
        text: str
    ) -> str:
        """Get section content including all child sections."""
        # Find all children
        child_sections = [s for s in all_sections if s.parent_id == section.section_id]
        
        if not child_sections:
            return text[section.start_position:section.end_position]
        
        # Find the end position including all children
        end_position = max([s.end_position for s in child_sections + [section]])
        
        return text[section.start_position:end_position]
    
    async def _chunk_large_hierarchical_section(
        self,
        content: str,
        section: DocumentSection,
        doc_structure: DocumentStructure
    ) -> List[TextChunk]:
        """Chunk large hierarchical sections while preserving structure."""
        # Use structure-aware splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=['\n\n\n', '\n\n', '\n', '. ', ' ', '']
        )
        
        text_chunks = splitter.split_text(content)
        chunks = []
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = TextChunk(
                content=chunk_text,
                chunk_id=f"hierarchical_{section.section_id}_part_{i+1}",
                metadata={
                    'chunk_type': 'hierarchical_section',
                    'parent_section_id': section.section_id,
                    'section_level': section.level,
                    'part_number': i + 1,
                    'total_parts': len(text_chunks),
                    'preserves_hierarchy': True
                },
                start_position=section.start_position + i * 200,  # Approximate
                end_position=section.start_position + (i + 1) * 200,
                document_name=doc_structure.document_id,
                length=len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks


# Global chunking service instance
_chunking_service: Optional[ChunkingService] = None


def get_chunking_service(config: Optional[ChunkingConfig] = None) -> ChunkingService:
    """
    Get the global chunking service instance.
    
    Args:
        config: Optional configuration (only used for first initialization)
        
    Returns:
        ChunkingService instance
    """
    global _chunking_service
    
    if _chunking_service is None:
        _chunking_service = ChunkingService(config)
    
    return _chunking_service
