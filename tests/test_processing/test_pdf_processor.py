"""
Unit tests for PDF processing components.
Tests PDFProcessor and OCRProcessor functionality with various document types.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from PIL import Image
import numpy as np

from processing.pdf_processor import (
    PDFProcessor, PDFProcessingConfig, DocumentType, ExtractionMethod,
    OCRConfig, ExtractedText, DocumentMetadata, ProcessedDocument
)
from processing.ocr_processor import (
    OCRProcessor, OCRConfig as OCRProcessorConfig, OCRQuality, 
    DocumentLayout, OCRResult
)
from core.exceptions import PDFExtractionError, OCRProcessingError


class TestPDFProcessor:
    """Test suite for PDFProcessor."""
    
    @pytest.fixture
    def pdf_processor(self):
        """Create PDFProcessor instance with test configuration."""
        config = PDFProcessingConfig(
            max_memory_mb=1024,  # Reduced for testing
            page_batch_size=5,
            timeout_seconds=30
        )
        return PDFProcessor(config)
    
    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Create a sample PDF file for testing."""
        # This would create a simple PDF file
        # For actual testing, you'd use a real PDF creation library
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"mock pdf content")  # Simplified for testing
        return pdf_path
    
    def test_init_configuration(self):
        """Test PDFProcessor initialization with custom configuration."""
        config = PDFProcessingConfig(
            max_memory_mb=2048,
            page_batch_size=20,
            timeout_seconds=60
        )
        processor = PDFProcessor(config)
        
        assert processor.config.max_memory_mb == 2048
        assert processor.config.page_batch_size == 20
        assert processor.config.timeout_seconds == 60
    
    def test_document_type_detection(self, pdf_processor):
        """Test document type detection logic."""
        metadata = DocumentMetadata()
        
        # Test table-heavy detection
        metadata.has_tables = True
        doc_type = pdf_processor._detect_document_type(metadata, 1000, 10)
        assert doc_type == DocumentType.TABLE_HEAVY
        
        # Test government notice detection
        metadata.has_tables = False
        metadata.title = "Government Notice - Policy Update"
        doc_type = pdf_processor._detect_document_type(metadata, 1000, 0)
        assert doc_type == DocumentType.GOVERNMENT_NOTICE
        
        # Test change request detection
        metadata.title = "Change Request CR-001"
        doc_type = pdf_processor._detect_document_type(metadata, 1000, 0)
        assert doc_type == DocumentType.CHANGE_REQUEST
        
        # Test manual detection
        metadata.title = "User Manual v2.0"
        metadata.page_count = 50
        doc_type = pdf_processor._detect_document_type(metadata, 1000, 0)
        assert doc_type == DocumentType.MANUAL
        
        # Test form letter detection
        metadata.title = "Letter to Customer"
        metadata.page_count = 2
        doc_type = pdf_processor._detect_document_type(metadata, 500, 0)
        assert doc_type == DocumentType.FORM_LETTER
    
    def test_table_detection_in_text(self, pdf_processor):
        """Test table detection heuristics."""
        # Text with tab characters (table-like)
        table_text = "Name\tAge\tCity\nJohn\t25\tNew York\nJane\t30\tBoston"
        assert pdf_processor._detect_tables_in_text(table_text) == True
        
        # Text with pipe separators
        pipe_text = "Name | Age | City\nJohn | 25 | New York\nJane | 30 | Boston"
        assert pdf_processor._detect_tables_in_text(pipe_text) == True
        
        # Regular paragraph text
        normal_text = "This is a regular paragraph with normal text flow and no table structure."
        assert pdf_processor._detect_tables_in_text(normal_text) == False
    
    @pytest.mark.asyncio
    async def test_process_pdf_file_not_found(self, pdf_processor):
        """Test handling of non-existent PDF files."""
        with pytest.raises(PDFExtractionError) as exc_info:
            await pdf_processor.process_pdf("nonexistent_file.pdf")
        
        assert "PDF file not found" in str(exc_info.value)
        assert exc_info.value.error_code == "PDF_EXTRACTION_ERROR"
    
    @pytest.mark.asyncio
    @patch('processing.pdf_processor.fitz')
    async def test_extract_metadata(self, mock_fitz, pdf_processor, sample_pdf_path):
        """Test PDF metadata extraction."""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_doc.metadata = {
            'title': 'Test Document',
            'author': 'Test Author',
            'subject': 'Test Subject',
            'creator': 'Test Creator'
        }
        mock_doc.page_count = 5
        mock_fitz.open.return_value = mock_doc
        
        metadata = await pdf_processor._extract_metadata(sample_pdf_path)
        
        assert metadata.title == 'Test Document'
        assert metadata.author == 'Test Author'
        assert metadata.page_count == 5
        assert isinstance(metadata.file_size_bytes, int)
    
    @pytest.mark.asyncio
    @patch('processing.pdf_processor.fitz')
    async def test_process_single_page_pymupdf(self, mock_fitz, pdf_processor):
        """Test single page processing with PyMuPDF."""
        # Mock document and page
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "This is extracted text from the PDF page."
        mock_page.get_images.return_value = []
        mock_doc.__getitem__.return_value = mock_page
        
        metadata = DocumentMetadata(detected_type=DocumentType.MIXED_CONTENT)
        
        result = await pdf_processor._process_single_page(mock_doc, 0, metadata)
        
        assert isinstance(result, ExtractedText)
        assert result.content == "This is extracted text from the PDF page."
        assert result.page_number == 1
        assert result.extraction_method == ExtractionMethod.PYMUPDF
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    @patch('processing.pdf_processor.fitz')
    async def test_process_single_page_with_ocr_fallback(self, mock_fitz, pdf_processor):
        """Test single page processing with OCR fallback for scanned content."""
        # Mock document and page with minimal text (triggering OCR)
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = ""  # No extractable text
        mock_page.get_images.return_value = [1, 2]  # Has images
        mock_doc.__getitem__.return_value = mock_page
        
        metadata = DocumentMetadata(detected_type=DocumentType.MIXED_CONTENT)
        
        # Mock OCR extraction
        with patch.object(pdf_processor, '_extract_with_ocr') as mock_ocr:
            mock_ocr.return_value = ("OCR extracted text", 0.85)
            
            result = await pdf_processor._process_single_page(mock_doc, 0, metadata)
            
            assert result.content == "OCR extracted text"
            assert result.extraction_method == ExtractionMethod.OCR_ONLY
            assert result.confidence == 0.85
    
    def test_memory_check_normal(self, pdf_processor):
        """Test memory usage check under normal conditions."""
        with patch('processing.pdf_processor.psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 512  # 512MB
            
            # Should not raise exception
            pdf_processor._check_memory_usage()
    
    def test_memory_check_exceeded(self, pdf_processor):
        """Test memory usage check when limit is exceeded."""
        with patch('processing.pdf_processor.psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 2048  # 2GB (exceeds 1GB limit)
            
            from core.exceptions import RAGMemoryError
            with pytest.raises(RAGMemoryError) as exc_info:
                pdf_processor._check_memory_usage()
            
            assert "Memory usage exceeded limit" in str(exc_info.value)
    
    def test_processing_stats_tracking(self, pdf_processor):
        """Test processing statistics tracking."""
        # Create mock processed document
        pages = [
            ExtractedText("Text 1", 1, ExtractionMethod.PYMUPDF, 1.0),
            ExtractedText("Text 2", 2, ExtractionMethod.OCR_ONLY, 0.8)
        ]
        doc = ProcessedDocument("test.pdf", DocumentMetadata(), pages)
        
        # Update stats
        pdf_processor._update_stats(doc, 5.0)
        
        stats = pdf_processor.get_processing_stats()
        assert stats['documents_processed'] == 1
        assert stats['pages_processed'] == 2
        assert stats['ocr_pages'] == 1
        assert stats['total_processing_time'] == 5.0
        assert stats['avg_processing_time'] == 5.0
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, pdf_processor):
        """Test health check when system is healthy."""
        with patch('processing.pdf_processor.pytesseract.get_tesseract_version') as mock_tesseract:
            mock_tesseract.return_value = "4.1.1"
            
            with patch('processing.pdf_processor.psutil.Process') as mock_process:
                mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
                
                health = await pdf_processor.health_check()
                
                assert health["status"] == "healthy"
                assert len(health["issues"]) == 0
                assert "memory_usage_mb" in health
                assert "stats" in health
    
    @pytest.mark.asyncio
    async def test_health_check_ocr_unavailable(self, pdf_processor):
        """Test health check when OCR is unavailable."""
        with patch('processing.pdf_processor.pytesseract.get_tesseract_version') as mock_tesseract:
            mock_tesseract.side_effect = Exception("Tesseract not found")
            
            health = await pdf_processor.health_check()
            
            assert health["status"] == "degraded"
            assert any("OCR not available" in issue for issue in health["issues"])


class TestOCRProcessor:
    """Test suite for OCRProcessor."""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create OCRProcessor instance with test configuration."""
        config = OCRProcessorConfig(
            quality_level=OCRQuality.BALANCED,
            confidence_threshold=0.7,
            enable_preprocessing=True
        )
        return OCRProcessor(config)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL image for testing."""
        return Image.new('RGB', (200, 100), color='white')
    
    @pytest.fixture
    def sample_text_image(self):
        """Create a sample image with text-like content."""
        # Create image with some pattern that resembles text
        img = Image.new('RGB', (400, 200), color='white')
        # In a real test, you'd draw actual text on the image
        return img
    
    def test_init_configuration(self):
        """Test OCRProcessor initialization."""
        config = OCRProcessorConfig(
            quality_level=OCRQuality.HIGH_QUALITY,
            confidence_threshold=0.8,
            languages=['eng', 'spa']
        )
        processor = OCRProcessor(config)
        
        assert processor.config.quality_level == OCRQuality.HIGH_QUALITY
        assert processor.config.confidence_threshold == 0.8
        assert processor.config.languages == ['eng', 'spa']
    
    def test_load_image_pil(self, ocr_processor, sample_image):
        """Test loading PIL Image."""
        loaded = ocr_processor._load_image(sample_image)
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (200, 100)
    
    def test_load_image_numpy(self, ocr_processor):
        """Test loading numpy array."""
        np_array = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White image
        loaded = ocr_processor._load_image(np_array)
        
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (200, 100)  # Note: PIL uses (width, height)
    
    def test_load_image_invalid_type(self, ocr_processor):
        """Test loading invalid image type."""
        from core.exceptions import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            ocr_processor._load_image("invalid_type")
        
        assert "Unsupported image type" in str(exc_info.value)
    
    def test_detect_layout_single_column(self, ocr_processor, sample_image):
        """Test single column layout detection."""
        layout = ocr_processor._detect_layout(sample_image)
        # With a simple white image, should default to single column
        assert layout in [DocumentLayout.SINGLE_COLUMN, DocumentLayout.MIXED]
    
    def test_detect_lines_horizontal(self, ocr_processor):
        """Test horizontal line detection."""
        # Create image with horizontal line
        img_array = np.ones((100, 200), dtype=np.uint8) * 255
        img_array[50:52, :] = 0  # Draw horizontal line
        
        lines = ocr_processor._detect_lines(img_array, horizontal=True)
        assert len(lines) > 0
    
    def test_detect_lines_vertical(self, ocr_processor):
        """Test vertical line detection."""
        # Create image with vertical line
        img_array = np.ones((100, 200), dtype=np.uint8) * 255
        img_array[:, 100:102] = 0  # Draw vertical line
        
        lines = ocr_processor._detect_lines(img_array, horizontal=False)
        assert len(lines) > 0
    
    def test_detect_columns_single(self, ocr_processor):
        """Test single column detection."""
        # Simple image should be detected as single column
        img_array = np.ones((100, 200), dtype=np.uint8) * 255
        columns = ocr_processor._detect_columns(img_array)
        assert columns >= 1
    
    def test_get_tesseract_config_quality_levels(self, ocr_processor):
        """Test Tesseract configuration for different quality levels."""
        # Test different quality levels
        fast_config = ocr_processor._get_tesseract_config(OCRQuality.FAST, DocumentLayout.SINGLE_COLUMN)
        balanced_config = ocr_processor._get_tesseract_config(OCRQuality.BALANCED, DocumentLayout.SINGLE_COLUMN)
        high_config = ocr_processor._get_tesseract_config(OCRQuality.HIGH_QUALITY, DocumentLayout.SINGLE_COLUMN)
        
        assert all(isinstance(config, str) for config in [fast_config, balanced_config, high_config])
        assert "--oem" in fast_config
        assert "--psm" in balanced_config
    
    def test_get_tesseract_config_layouts(self, ocr_processor):
        """Test Tesseract configuration for different layouts."""
        table_config = ocr_processor._get_tesseract_config(OCRQuality.BALANCED, DocumentLayout.TABLE)
        form_config = ocr_processor._get_tesseract_config(OCRQuality.BALANCED, DocumentLayout.FORM)
        multi_col_config = ocr_processor._get_tesseract_config(OCRQuality.BALANCED, DocumentLayout.MULTI_COLUMN)
        
        assert "--psm 6" in table_config
        assert "--psm 6" in form_config
        assert "--psm 2" in multi_col_config
    
    @patch('processing.ocr_processor.pytesseract.image_to_data')
    def test_process_ocr_data(self, mock_image_to_data, ocr_processor):
        """Test processing of OCR data from Tesseract."""
        # Mock Tesseract output
        mock_ocr_data = {
            'text': ['', 'Hello', 'world', 'test', ''],
            'conf': ['-1', '95', '87', '92', '-1']
        }
        
        result = ocr_processor._process_ocr_data(mock_ocr_data)
        
        assert isinstance(result, OCRResult)
        assert result.text == "Hello world test"
        assert result.word_count == 3
        assert result.confidence > 0.8  # Should be high confidence
    
    def test_detect_language_english(self, ocr_processor):
        """Test English language detection."""
        text = "This is a sample text with common English words like the and and."
        languages = ocr_processor._detect_language(text)
        assert 'eng' in languages
    
    @patch('processing.ocr_processor.pytesseract.image_to_data')
    @pytest.mark.asyncio
    async def test_process_image_success(self, mock_image_to_data, ocr_processor, sample_image):
        """Test successful image processing."""
        # Mock successful OCR result
        mock_image_to_data.return_value = {
            'text': ['Sample', 'text', 'extracted'],
            'conf': ['90', '85', '88']
        }
        
        result = await ocr_processor.process_image(sample_image)
        
        assert isinstance(result, OCRResult)
        assert len(result.text) > 0
        assert result.confidence > 0
        assert result.processing_time >= 0
    
    @patch('processing.ocr_processor.pytesseract.image_to_data')
    @pytest.mark.asyncio
    async def test_process_image_with_fallbacks(self, mock_image_to_data, ocr_processor, sample_image):
        """Test image processing with fallback strategies."""
        # Mock low-confidence initial result to trigger fallbacks
        mock_image_to_data.side_effect = [
            {  # Initial result (low confidence)
                'text': ['poor', 'quality'],
                'conf': ['30', '25']  # Low confidence
            },
            {  # Fallback result (better)
                'text': ['better', 'quality', 'text'],
                'conf': ['85', '90', '88']  # Higher confidence
            }
        ]
        
        # Configure for fallbacks
        ocr_processor.config.confidence_threshold = 0.7
        ocr_processor.config.enable_fallbacks = True
        
        result = await ocr_processor.process_image(sample_image)
        
        assert result.fallback_used == True
        # Should use the better result
        assert "better quality text" in result.text or len(result.text) > 0
    
    def test_stats_tracking(self, ocr_processor):
        """Test OCR processing statistics tracking."""
        # Create mock results
        result1 = OCRResult("Text 1", 0.8, 1.5)
        result2 = OCRResult("Text 2", 0.9, 2.0)
        
        ocr_processor._update_stats(result1)
        ocr_processor._update_stats(result2)
        
        stats = ocr_processor.get_stats()
        assert stats['pages_processed'] == 2
        assert stats['avg_confidence'] == 0.85
        assert len(stats['processing_times']) == 2
        assert stats['avg_processing_time'] == 1.75
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, ocr_processor):
        """Test health check when OCR processor is healthy."""
        with patch.object(ocr_processor, 'process_image') as mock_process:
            mock_process.return_value = OCRResult("test", 0.8, 1.0)
            
            health = await ocr_processor.health_check()
            
            assert health["status"] == "healthy"
            assert len(health["issues"]) == 0
            assert "stats" in health
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, ocr_processor):
        """Test health check when OCR processor is unhealthy."""
        with patch.object(ocr_processor, 'process_image') as mock_process:
            mock_process.side_effect = Exception("OCR failed")
            
            health = await ocr_processor.health_check()
            
            assert health["status"] == "unhealthy"
            assert len(health["issues"]) > 0


class TestIntegration:
    """Integration tests for PDF processing components."""
    
    @pytest.mark.asyncio
    async def test_pdf_processor_with_ocr_integration(self):
        """Test PDF processor integration with OCR processor."""
        pdf_config = PDFProcessingConfig(
            max_memory_mb=1024,
            page_batch_size=5
        )
        pdf_processor = PDFProcessor(pdf_config)
        
        # Mock a scanned PDF scenario
        with patch('processing.pdf_processor.fitz') as mock_fitz:
            mock_doc = Mock()
            mock_doc.page_count = 2
            mock_doc.metadata = {'title': 'Test Document'}
            
            # Mock pages with minimal text (simulating scanned pages)
            mock_page1 = Mock()
            mock_page1.get_text.return_value = ""  # No text
            mock_page1.get_images.return_value = [1]  # Has image
            
            mock_page2 = Mock()
            mock_page2.get_text.return_value = "Some extractable text"
            mock_page2.get_images.return_value = []
            
            mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
            mock_fitz.open.return_value = mock_doc
            
            # Mock OCR processing
            with patch.object(pdf_processor, '_extract_with_ocr') as mock_ocr:
                mock_ocr.return_value = ("OCR extracted text from page 1", 0.85)
                
                # Create a temporary file path
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    tmp_file.write(b"mock pdf content")
                
                try:
                    # Process the PDF
                    result = await pdf_processor.process_pdf(tmp_path)
                    
                    assert isinstance(result, ProcessedDocument)
                    assert result.total_pages == 2
                    assert result.ocr_pages == 1  # One page required OCR
                    assert len(result.pages) == 2
                    
                    # Verify OCR was used for first page
                    assert result.pages[0].extraction_method == ExtractionMethod.OCR_ONLY
                    assert result.pages[0].content == "OCR extracted text from page 1"
                    
                    # Verify PyMuPDF was used for second page
                    assert result.pages[1].extraction_method == ExtractionMethod.PYMUPDF
                    assert result.pages[1].content == "Some extractable text"
                    
                finally:
                    # Clean up
                    tmp_path.unlink(missing_ok=True)


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests for PDF processing."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_batch_processing(self):
        """Test processing large batches of pages."""
        config = PDFProcessingConfig(
            page_batch_size=50,  # Large batch
            max_memory_mb=2048
        )
        processor = PDFProcessor(config)
        
        # Mock a large document
        with patch('processing.pdf_processor.fitz') as mock_fitz:
            mock_doc = Mock()
            mock_doc.page_count = 100  # Large document
            
            # Mock pages
            mock_pages = []
            for i in range(100):
                mock_page = Mock()
                mock_page.get_text.return_value = f"Text from page {i + 1}"
                mock_page.get_images.return_value = []
                mock_pages.append(mock_page)
            
            mock_doc.__getitem__.side_effect = mock_pages
            mock_fitz.open.return_value = mock_doc
            
            # Test that large document processing completes without errors
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(b"mock large pdf content")
            
            try:
                result = await processor.process_pdf(tmp_path)
                
                assert result.total_pages == 100
                assert result.successful_pages == 100
                assert len(result.pages) == 100
                
            finally:
                tmp_path.unlink(missing_ok=True)
    
    def test_memory_monitoring(self):
        """Test memory usage monitoring under various conditions."""
        config = PDFProcessingConfig(max_memory_mb=100)  # Low limit for testing
        processor = PDFProcessor(config)
        
        # Test normal memory usage
        with patch('processing.pdf_processor.psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 50 * 1024 * 1024  # 50MB
            processor._check_memory_usage()  # Should not raise
        
        # Test memory limit exceeded
        with patch('processing.pdf_processor.psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 200 * 1024 * 1024  # 200MB
            
            from core.exceptions import RAGMemoryError
            with pytest.raises(RAGMemoryError):
                processor._check_memory_usage()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
