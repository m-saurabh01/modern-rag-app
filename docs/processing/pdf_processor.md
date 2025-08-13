# PDF Processor Documentation

## Overview

The `PDFProcessor` class provides a comprehensive solution for extracting text and metadata from PDF documents using multiple extraction engines. It's designed to handle diverse PDF types including text-based, scanned, and hybrid documents while maintaining memory efficiency for large document processing.

## Architecture

The PDF processor implements a multi-engine extraction strategy with intelligent fallback mechanisms:

- **Primary Engine**: PyMuPDF (fitz) for fast text extraction
- **Fallback Engine**: pdfplumber for complex layouts and table extraction
- **OCR Integration**: Seamless integration with OCRProcessor for scanned content
- **Memory Management**: Page-by-page processing to handle large documents

## Class Structure

```python
class PDFProcessor:
    def __init__(self, ocr_processor: Optional[OCRProcessor] = None)
```

## Methods Documentation

### `__init__(self, ocr_processor: Optional[OCRProcessor] = None)`

**Purpose**: Initializes the PDF processor with optional OCR capabilities.

**Parameters**:
- `ocr_processor` (Optional[OCRProcessor]): OCR processor instance for handling scanned pages

**What it achieves**:
- Sets up the PDF processor with configurable OCR integration
- Enables flexible processing pipeline based on document types
- Provides foundation for multi-engine text extraction

**Usage Example**:
```python
from processing.ocr_processor import OCRProcessor

ocr = OCRProcessor()
pdf_processor = PDFProcessor(ocr_processor=ocr)
```

### `extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Any]`

**Purpose**: Main entry point for extracting text and metadata from PDF documents.

**Parameters**:
- `pdf_path` (Union[str, Path]): Path to the PDF file to process

**Returns**:
- `Dict[str, Any]`: Comprehensive extraction results containing:
  - `text`: Extracted text content
  - `metadata`: Document metadata (title, author, pages, etc.)
  - `pages`: List of individual page data
  - `extraction_method`: Method used for extraction
  - `has_scanned_pages`: Boolean indicating presence of scanned content

**What it achieves**:
- Orchestrates the entire PDF processing pipeline
- Determines optimal extraction strategy based on document characteristics
- Provides comprehensive results with metadata and processing information
- Handles errors gracefully with detailed error reporting

**Processing Flow**:
1. Opens PDF document using PyMuPDF
2. Extracts basic metadata (title, author, page count)
3. Processes pages individually for memory efficiency
4. Detects scanned pages and applies appropriate extraction method
5. Consolidates results with comprehensive metadata

**Usage Example**:
```python
result = pdf_processor.extract_text_from_pdf("document.pdf")
print(f"Extracted {len(result['text'])} characters")
print(f"Document has {result['metadata']['pages']} pages")
```

### `_extract_page_text(self, page: fitz.Page) -> Dict[str, Any]`

**Purpose**: Extracts text content from a single PDF page using the primary extraction engine.

**Parameters**:
- `page` (fitz.Page): PyMuPDF page object to extract text from

**Returns**:
- `Dict[str, Any]`: Page extraction results containing:
  - `text`: Extracted text content
  - `method`: Extraction method used ("pymupdf")
  - `page_number`: Page number (0-indexed)
  - `char_count`: Number of characters extracted

**What it achieves**:
- Performs fast text extraction using PyMuPDF's text() method
- Provides page-level text extraction with metadata
- Serves as the primary extraction method for text-based PDFs
- Returns structured data for pipeline processing

**Internal Processing**:
- Uses PyMuPDF's optimized text extraction
- Handles empty pages gracefully
- Provides character count for quality assessment
- Maintains page number tracking for document structure

### `_extract_page_text_fallback(self, pdf_path: Union[str, Path], page_num: int) -> Dict[str, Any]`

**Purpose**: Alternative text extraction method using pdfplumber for complex layouts and table structures.

**Parameters**:
- `pdf_path` (Union[str, Path]): Path to the PDF file
- `page_num` (int): Page number to extract (0-indexed)

**Returns**:
- `Dict[str, Any]`: Page extraction results with fallback method metadata

**What it achieves**:
- Provides robust extraction for complex PDF layouts
- Handles tables, columns, and structured content better than PyMuPDF
- Serves as fallback when primary extraction fails or produces poor results
- Maintains consistency with primary extraction interface

**Use Cases**:
- Complex multi-column layouts
- Documents with embedded tables
- PDFs with non-standard text positioning
- When primary extraction yields insufficient text

**Processing Approach**:
- Opens PDF using pdfplumber for specific page
- Extracts text using pdfplumber's layout-aware methods
- Handles extraction errors with graceful fallback
- Returns consistent data structure

### `_is_page_scanned(self, page: fitz.Page) -> bool`

**Purpose**: Determines if a PDF page contains primarily scanned/image content requiring OCR processing.

**Parameters**:
- `page` (fitz.Page): PyMuPDF page object to analyze

**Returns**:
- `bool`: True if page appears to be scanned, False otherwise

**What it achieves**:
- Implements intelligent document type detection
- Enables automatic OCR triggering for scanned pages
- Optimizes processing pipeline by avoiding unnecessary OCR
- Provides foundation for hybrid document handling

**Detection Algorithm**:
1. **Text Content Analysis**: Counts extractable text characters
2. **Image Presence Check**: Detects embedded images on the page
3. **Threshold Evaluation**: Applies heuristics to determine scan status:
   - Pages with < 50 characters and images likely scanned
   - Pure image pages without text classified as scanned
   - Text-heavy pages classified as digital

**Heuristics Used**:
- Character count threshold (50 characters)
- Image-to-text ratio analysis
- Page dimension consideration
- Content density evaluation

### `_extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]`

**Purpose**: Extracts comprehensive metadata from PDF documents for indexing and organization.

**Parameters**:
- `doc` (fitz.Document): PyMuPDF document object

**Returns**:
- `Dict[str, Any]`: Document metadata containing:
  - `title`: Document title
  - `author`: Document author
  - `subject`: Document subject
  - `creator`: Application that created the PDF
  - `producer`: PDF producer application
  - `creation_date`: Document creation timestamp
  - `modification_date`: Last modification timestamp
  - `pages`: Total page count
  - `file_size`: File size information (when available)

**What it achieves**:
- Provides comprehensive document information for RAG indexing
- Enables document organization and filtering capabilities
- Supports document provenance and version tracking
- Facilitates search and retrieval optimization

**Metadata Extraction Process**:
1. Accesses PyMuPDF's metadata dictionary
2. Extracts standard PDF metadata fields
3. Handles missing or malformed metadata gracefully
4. Provides consistent metadata structure across documents
5. Includes computed fields like page count

## Error Handling

The PDF processor implements comprehensive error handling:

### Common Error Scenarios
- **File Access Errors**: Invalid file paths, permission issues
- **Corrupted PDF Files**: Malformed or damaged PDF documents
- **Memory Constraints**: Large documents exceeding available memory
- **Extraction Failures**: Text extraction engine failures

### Error Recovery Strategies
- **Graceful Degradation**: Falls back to alternative extraction methods
- **Partial Processing**: Continues processing remaining pages when individual pages fail
- **Detailed Error Logging**: Provides comprehensive error information for debugging
- **Resource Cleanup**: Ensures proper resource management even during failures

## Performance Considerations

### Memory Management
- **Page-by-Page Processing**: Processes large documents without loading entire content into memory
- **Resource Cleanup**: Properly closes PDF documents and releases resources
- **Batch Processing Support**: Designed for processing multiple documents efficiently

### Processing Optimization
- **Engine Selection**: Uses fastest appropriate extraction method
- **OCR Triggering**: Only applies OCR when necessary based on content analysis
- **Caching Strategy**: Avoids redundant processing of already analyzed content

### Scalability Features
- **Async-Ready Design**: Compatible with asynchronous processing frameworks
- **Memory-Efficient**: Suitable for processing large document collections
- **Configurable Thresholds**: Allows tuning for different hardware configurations

## Integration Points

### OCR Processor Integration
```python
# With OCR support
ocr_processor = OCRProcessor()
pdf_processor = PDFProcessor(ocr_processor=ocr_processor)

# Automatic OCR for scanned pages
result = pdf_processor.extract_text_from_pdf("scanned_document.pdf")
```

### Embedding Service Integration
```python
# Process and embed documents
pdf_result = pdf_processor.extract_text_from_pdf("document.pdf")
embeddings = embedding_service.generate_embeddings([pdf_result['text']])
```

### Chunking Service Integration
```python
# Extract and chunk document content
pdf_result = pdf_processor.extract_text_from_pdf("document.pdf")
chunks = chunking_service.chunk_text(pdf_result['text'])
```

## Usage Examples

### Basic Text Extraction
```python
processor = PDFProcessor()
result = processor.extract_text_from_pdf("document.pdf")
print(f"Extracted: {result['text'][:200]}...")
```

### Processing with OCR Support
```python
ocr = OCRProcessor()
processor = PDFProcessor(ocr_processor=ocr)
result = processor.extract_text_from_pdf("mixed_document.pdf")

if result['has_scanned_pages']:
    print("Document contained scanned pages processed with OCR")
```

### Batch Processing
```python
import os
from pathlib import Path

processor = PDFProcessor()
pdf_files = Path("documents/").glob("*.pdf")

for pdf_file in pdf_files:
    try:
        result = processor.extract_text_from_pdf(pdf_file)
        print(f"Processed {pdf_file.name}: {len(result['text'])} characters")
    except Exception as e:
        print(f"Failed to process {pdf_file.name}: {e}")
```

## Configuration and Tuning

### Scanned Page Detection Tuning
The `_is_page_scanned` method uses configurable thresholds:
- **Character Threshold**: Minimum characters for digital page classification (default: 50)
- **Image Ratio**: Image-to-content ratio for scan detection
- **Page Density**: Content density analysis for classification

### Memory Management Settings
- **Page Processing**: Individual page processing prevents memory overflow
- **Resource Limits**: Automatic cleanup of PDF document resources
- **Batch Size**: Configurable batch processing for multiple documents

### Performance Optimization
- **Engine Selection**: Choose primary extraction engine based on document types
- **OCR Threshold**: Configure when to trigger OCR processing
- **Fallback Strategy**: Define fallback behavior for extraction failures

This PDF processor provides a robust, scalable solution for text extraction from diverse PDF document types while maintaining memory efficiency and providing comprehensive error handling.