"""
PDF processing service for the Modern RAG application.
Handles multi-format PDF text extraction with OCR integration for scanned documents.
Optimized for diverse document types: tables, notices, letters, manuals, CR documents, govt notices.
"""

import asyncio
import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import psutil
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np

from core.exceptions import (
    PDFExtractionError,
    OCRProcessingError,
    RAGMemoryError,
    ValidationError
)
from config.settings import get_settings


class DocumentType(Enum):
    """Detected document types for specialized processing."""
    TABLE_HEAVY = "table_heavy"
    FORM_LETTER = "form_letter"
    MANUAL = "manual"
    GOVERNMENT_NOTICE = "government_notice"
    CHANGE_REQUEST = "change_request"
    MIXED_CONTENT = "mixed_content"
    UNKNOWN = "unknown"


class OCREngine(Enum):
    """Available OCR engines."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"  # Future implementation
    HYBRID = "hybrid"


class ExtractionMethod(Enum):
    """PDF text extraction methods."""
    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"
    OCR_ONLY = "ocr_only"
    HYBRID = "hybrid"


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    engine: OCREngine = OCREngine.TESSERACT
    confidence_threshold: float = 0.7
    languages: List[str] = field(default_factory=lambda: ['eng'])
    dpi: int = 300
    preprocessing: bool = True
    fallback_enabled: bool = True
    
    # Tesseract-specific settings
    tesseract_config: str = '--oem 3 --psm 6'
    
    # Quality thresholds
    min_confidence: float = 0.5
    retry_threshold: float = 0.6


@dataclass
class PDFProcessingConfig:
    """Configuration for PDF processing."""
    extraction_method: ExtractionMethod = ExtractionMethod.HYBRID
    ocr_config: OCRConfig = field(default_factory=OCRConfig)
    
    # Memory management
    max_memory_mb: int = 4096  # 4GB max per document
    page_batch_size: int = 10
    
    # Quality settings
    min_text_length: int = 50
    scanned_threshold: float = 0.1  # Ratio of extractable text to determine if scanned
    
    # Performance settings
    max_workers: int = 4
    timeout_seconds: int = 300
    
    # Document type detection
    enable_type_detection: bool = True
    table_detection_threshold: int = 5  # Min tables to classify as table-heavy


@dataclass
class ExtractedText:
    """Container for extracted text with metadata."""
    content: str
    page_number: int
    extraction_method: ExtractionMethod
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    word_count: int = 0
    character_count: int = 0
    line_count: int = 0
    
    def __post_init__(self):
        """Calculate text statistics."""
        if self.content:
            self.word_count = len(self.content.split())
            self.character_count = len(self.content)
            self.line_count = len(self.content.splitlines())


@dataclass
class DocumentMetadata:
    """PDF document metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    file_size_bytes: int = 0
    detected_type: DocumentType = DocumentType.UNKNOWN
    has_images: bool = False
    has_tables: bool = False
    is_scanned: bool = False
    processing_time_seconds: float = 0


@dataclass
class ProcessedDocument:
    """Complete processed document with all extracted content."""
    file_path: str
    metadata: DocumentMetadata
    pages: List[ExtractedText]
    full_text: str = ""
    
    # Processing statistics
    total_pages: int = 0
    successful_pages: int = 0
    ocr_pages: int = 0
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.total_pages = len(self.pages)
        self.successful_pages = len([p for p in self.pages if p.content.strip()])
        self.ocr_pages = len([p for p in self.pages if p.extraction_method == ExtractionMethod.OCR_ONLY])
        self.full_text = "\n\n".join([page.content for page in self.pages if page.content.strip()])


class PDFProcessor:
    """
    Advanced PDF processing service with OCR integration.
    
    Handles diverse document types with intelligent extraction strategies:
    - Multi-engine text extraction (PyMuPDF, pdfplumber)
    - OCR integration with confidence-based fallbacks
    - Memory-efficient page-by-page processing
    - Document type detection and specialized handling
    - Progress tracking and error recovery
    """
    
    def __init__(self, config: Optional[PDFProcessingConfig] = None):
        """Initialize PDF processor with configuration."""
        self.config = config or PDFProcessingConfig()
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Performance tracking
        self.stats = {
            'documents_processed': 0,
            'pages_processed': 0,
            'ocr_pages': 0,
            'extraction_errors': 0,
            'total_processing_time': 0.0
        }
        
        # Validate OCR availability
        self._validate_ocr_setup()
    
    def _validate_ocr_setup(self) -> None:
        """Validate OCR engine availability."""
        try:
            # Test Tesseract availability
            pytesseract.get_tesseract_version()
            self.logger.info("Tesseract OCR engine available")
        except Exception as e:
            self.logger.warning(f"Tesseract not available: {e}")
            if self.config.ocr_config.engine == OCREngine.TESSERACT:
                raise OCRProcessingError(
                    message="Tesseract OCR engine not available but required",
                    details={"error": str(e)}
                )
    
    async def process_pdf(self, file_path: Union[str, Path]) -> ProcessedDocument:
        """
        Process a single PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ProcessedDocument with extracted content and metadata
            
        Raises:
            PDFExtractionError: If PDF processing fails
            MemoryError: If memory limits are exceeded
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise PDFExtractionError(
                message=f"PDF file not found: {file_path}",
                pdf_path=str(file_path)
            )
        
        start_time = time.time()
        
        try:
            # Memory check before processing
            self._check_memory_usage()
            
            self.logger.info(f"Starting PDF processing: {file_path}")
            
            # Initialize document metadata
            metadata = await self._extract_metadata(file_path)
            
            # Process document pages
            pages = await self._process_pages(file_path, metadata)
            
            # Create processed document
            processed_doc = ProcessedDocument(
                file_path=str(file_path),
                metadata=metadata,
                pages=pages
            )
            
            # Update processing statistics
            processing_time = time.time() - start_time
            metadata.processing_time_seconds = processing_time
            
            self._update_stats(processed_doc, processing_time)
            
            self.logger.info(
                f"PDF processing completed: {file_path} "
                f"({processed_doc.successful_pages}/{processed_doc.total_pages} pages, "
                f"{processing_time:.2f}s)"
            )
            
            return processed_doc
            
        except Exception as e:
            if isinstance(e, (PDFExtractionError, OCRProcessingError, RAGMemoryError)):
                raise
            
            raise PDFExtractionError(
                message=f"Unexpected error processing PDF: {str(e)}",
                pdf_path=str(file_path),
                cause=e
            )
    
    async def _extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract PDF metadata and perform document analysis."""
        try:
            # Get file statistics
            file_stats = file_path.stat()
            
            # Extract PDF metadata using PyMuPDF
            doc = fitz.open(str(file_path))
            pdf_metadata = doc.metadata
            
            metadata = DocumentMetadata(
                title=pdf_metadata.get('title'),
                author=pdf_metadata.get('author'),
                subject=pdf_metadata.get('subject'),
                creator=pdf_metadata.get('creator'),
                producer=pdf_metadata.get('producer'),
                creation_date=pdf_metadata.get('creationDate'),
                modification_date=pdf_metadata.get('modDate'),
                page_count=doc.page_count,
                file_size_bytes=file_stats.st_size
            )
            
            # Document analysis
            await self._analyze_document_structure(doc, metadata)
            
            doc.close()
            
            return metadata
            
        except Exception as e:
            raise PDFExtractionError(
                message=f"Failed to extract PDF metadata: {str(e)}",
                pdf_path=str(file_path),
                cause=e
            )
    
    async def _analyze_document_structure(self, doc: fitz.Document, metadata: DocumentMetadata) -> None:
        """Analyze document structure and detect document type."""
        try:
            # Sample first few pages for analysis
            sample_pages = min(3, doc.page_count)
            
            table_count = 0
            image_count = 0
            total_text_length = 0
            
            for page_num in range(sample_pages):
                page = doc[page_num]
                
                # Check for images
                image_list = page.get_images()
                image_count += len(image_list)
                
                # Try to extract text to assess if scanned
                text = page.get_text()
                total_text_length += len(text.strip())
                
                # Basic table detection (look for grid-like structures)
                # This is a simplified heuristic - could be enhanced
                if self._detect_tables_in_text(text):
                    table_count += 1
            
            # Set metadata flags
            metadata.has_images = image_count > 0
            metadata.has_tables = table_count >= self.config.table_detection_threshold
            
            # Determine if document is likely scanned
            avg_text_per_page = total_text_length / sample_pages if sample_pages > 0 else 0
            metadata.is_scanned = avg_text_per_page < (metadata.page_count * 100 * self.config.scanned_threshold)
            
            # Detect document type
            metadata.detected_type = self._detect_document_type(metadata, total_text_length, table_count)
            
        except Exception as e:
            self.logger.warning(f"Document analysis failed: {e}")
            # Continue processing even if analysis fails
    
    def _detect_tables_in_text(self, text: str) -> bool:
        """Simple table detection based on text patterns."""
        # Look for common table indicators
        table_indicators = [
            '\t',  # Tab characters
            '|',   # Pipe separators
            '  ',  # Multiple spaces (common in aligned text)
        ]
        
        lines = text.split('\n')
        aligned_lines = 0
        
        for line in lines:
            if any(indicator in line for indicator in table_indicators):
                aligned_lines += 1
        
        # If more than 20% of lines show table-like structure
        return len(lines) > 0 and (aligned_lines / len(lines)) > 0.2
    
    def _detect_document_type(self, metadata: DocumentMetadata, text_length: int, table_count: int) -> DocumentType:
        """Detect document type based on analysis."""
        # Table-heavy documents
        if table_count >= self.config.table_detection_threshold or metadata.has_tables:
            return DocumentType.TABLE_HEAVY
        
        # Government notices (heuristic based on metadata)
        if metadata.title and any(keyword in metadata.title.lower() for keyword in 
                                ['notice', 'government', 'official', 'regulation', 'policy']):
            return DocumentType.GOVERNMENT_NOTICE
        
        # Change request documents
        if metadata.title and any(keyword in metadata.title.lower() for keyword in 
                                ['change request', 'cr', 'modification', 'amendment']):
            return DocumentType.CHANGE_REQUEST
        
        # User manuals (heuristic based on page count and structure)
        if metadata.page_count > 20 and metadata.title and any(keyword in metadata.title.lower() for keyword in 
                                                              ['manual', 'guide', 'handbook', 'documentation']):
            return DocumentType.MANUAL
        
        # Form letters (typically shorter)
        if metadata.page_count <= 3:
            return DocumentType.FORM_LETTER
        
        return DocumentType.MIXED_CONTENT
    
    async def _process_pages(self, file_path: Path, metadata: DocumentMetadata) -> List[ExtractedText]:
        """Process all pages of the PDF document."""
        try:
            # Open document
            doc = fitz.open(str(file_path))
            
            all_pages = []
            
            # Process pages in batches to manage memory
            total_pages = doc.page_count
            
            for batch_start in range(0, total_pages, self.config.page_batch_size):
                batch_end = min(batch_start + self.config.page_batch_size, total_pages)
                
                # Process batch
                batch_pages = await self._process_page_batch(
                    doc, batch_start, batch_end, metadata
                )
                
                all_pages.extend(batch_pages)
                
                # Memory check between batches
                self._check_memory_usage()
                
                self.logger.debug(f"Processed pages {batch_start + 1}-{batch_end} of {total_pages}")
            
            doc.close()
            
            return all_pages
            
        except Exception as e:
            raise PDFExtractionError(
                message=f"Failed to process PDF pages: {str(e)}",
                pdf_path=str(file_path),
                cause=e
            )
    
    async def _process_page_batch(
        self, 
        doc: fitz.Document, 
        start_page: int, 
        end_page: int,
        metadata: DocumentMetadata
    ) -> List[ExtractedText]:
        """Process a batch of pages."""
        tasks = []
        
        for page_num in range(start_page, end_page):
            task = asyncio.create_task(
                self._process_single_page(doc, page_num, metadata)
            )
            tasks.append(task)
        
        # Wait for all pages in batch to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_pages = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                page_num = start_page + i
                self.logger.error(f"Failed to process page {page_num + 1}: {result}")
                
                # Create empty page result
                processed_pages.append(ExtractedText(
                    content="",
                    page_number=page_num + 1,
                    extraction_method=ExtractionMethod.HYBRID,
                    confidence=0.0,
                    metadata={"error": str(result)}
                ))
            else:
                processed_pages.append(result)
        
        return processed_pages
    
    async def _process_single_page(
        self, 
        doc: fitz.Document, 
        page_num: int, 
        metadata: DocumentMetadata
    ) -> ExtractedText:
        """Process a single page with hybrid extraction strategy."""
        try:
            page = doc[page_num]
            
            # Try PyMuPDF extraction first
            text = page.get_text()
            extraction_method = ExtractionMethod.PYMUPDF
            confidence = 1.0 if text.strip() else 0.0
            
            # If minimal text found, try pdfplumber
            if len(text.strip()) < self.config.min_text_length:
                try:
                    plumber_text = await self._extract_with_pdfplumber(doc, page_num)
                    if len(plumber_text.strip()) > len(text.strip()):
                        text = plumber_text
                        extraction_method = ExtractionMethod.PDFPLUMBER
                        confidence = 0.8
                except Exception as e:
                    self.logger.debug(f"pdfplumber extraction failed for page {page_num + 1}: {e}")
            
            # If still minimal text and likely scanned, use OCR
            if (len(text.strip()) < self.config.min_text_length or 
                confidence < self.config.ocr_config.retry_threshold):
                
                try:
                    ocr_text, ocr_confidence = await self._extract_with_ocr(page)
                    if ocr_confidence > confidence and len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        extraction_method = ExtractionMethod.OCR_ONLY
                        confidence = ocr_confidence
                except Exception as e:
                    self.logger.debug(f"OCR extraction failed for page {page_num + 1}: {e}")
            
            # Create extracted text object
            extracted = ExtractedText(
                content=text,
                page_number=page_num + 1,
                extraction_method=extraction_method,
                confidence=confidence,
                metadata={
                    "document_type": metadata.detected_type.value,
                    "has_images": len(page.get_images()) > 0
                }
            )
            
            return extracted
            
        except Exception as e:
            raise PDFExtractionError(
                message=f"Failed to process page {page_num + 1}",
                pdf_path="",  # Will be filled by caller
                page_number=page_num + 1,
                cause=e
            )
    
    async def _extract_with_pdfplumber(self, doc: fitz.Document, page_num: int) -> str:
        """Extract text using pdfplumber for better table/form handling."""
        # This is a simplified implementation
        # In practice, you'd need to coordinate between PyMuPDF and pdfplumber
        # For now, we'll use PyMuPDF as the primary method
        return ""
    
    async def _extract_with_ocr(self, page: fitz.Page) -> Tuple[str, float]:
        """Extract text using OCR with confidence scoring."""
        try:
            # Convert page to image
            mat = fitz.Matrix(self.config.ocr_config.dpi / 72, self.config.ocr_config.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Preprocessing if enabled
            if self.config.ocr_config.preprocessing:
                image = self._preprocess_image(image)
            
            # Run OCR
            custom_config = self.config.ocr_config.tesseract_config
            
            # Extract text with confidence
            data = pytesseract.image_to_data(
                image,
                config=custom_config,
                lang='+'.join(self.config.ocr_config.languages),
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate confidence and extract text
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            text = ' '.join([
                data['text'][i] for i in range(len(data['text'])) 
                if int(data['conf'][i]) > self.config.ocr_config.min_confidence * 100
            ])
            
            return text, avg_confidence / 100.0
            
        except Exception as e:
            raise OCRProcessingError(
                message=f"OCR processing failed: {str(e)}",
                cause=e
            )
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL
            processed_image = Image.fromarray(binary)
            
            return processed_image
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image  # Return original if preprocessing fails
    
    def _check_memory_usage(self) -> None:
        """Check current memory usage and raise error if limit exceeded."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.config.max_memory_mb:
                raise RAGMemoryError(
                    message=f"Memory usage exceeded limit: {memory_mb:.1f}MB > {self.config.max_memory_mb}MB",
                    memory_usage_mb=memory_mb
                )
                
        except psutil.NoSuchProcess:
            # Process monitoring not available, continue
            pass
    
    def _update_stats(self, doc: ProcessedDocument, processing_time: float) -> None:
        """Update processing statistics."""
        self.stats['documents_processed'] += 1
        self.stats['pages_processed'] += doc.total_pages
        self.stats['ocr_pages'] += doc.ocr_pages
        self.stats['extraction_errors'] += len(doc.errors)
        self.stats['total_processing_time'] += processing_time
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.stats.copy()
        
        if stats['documents_processed'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['documents_processed']
            stats['avg_pages_per_doc'] = stats['pages_processed'] / stats['documents_processed']
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the PDF processor."""
        health = {
            "status": "healthy",
            "issues": []
        }
        
        try:
            # Check OCR availability
            pytesseract.get_tesseract_version()
        except Exception as e:
            health["status"] = "degraded"
            health["issues"].append(f"OCR not available: {e}")
        
        # Check memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            health["memory_usage_mb"] = memory_mb
            
            if memory_mb > self.config.max_memory_mb * 0.8:  # 80% threshold
                health["issues"].append(f"High memory usage: {memory_mb:.1f}MB")
        except Exception as e:
            health["issues"].append(f"Memory monitoring failed: {e}")
        
        # Add processing statistics
        health["stats"] = self.get_processing_stats()
        
        return health
