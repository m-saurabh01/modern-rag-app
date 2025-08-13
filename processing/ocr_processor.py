"""
OCR integration module for the Modern RAG application.
Provides hybrid OCR processing with confidence-based fallbacks and quality assessment.
Optimized for diverse document types with table and form handling.
"""

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

from core.exceptions import OCRProcessingError, ValidationError


class OCRQuality(Enum):
    """OCR quality levels."""
    FAST = "fast"           # Speed optimized
    BALANCED = "balanced"   # Balanced speed/quality
    HIGH_QUALITY = "high_quality"  # Quality optimized
    ADAPTIVE = "adaptive"   # Adaptive based on content


class DocumentLayout(Enum):
    """Document layout types for OCR optimization."""
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE = "table"
    FORM = "form"
    MIXED = "mixed"


@dataclass
class OCRResult:
    """OCR processing result with confidence and metadata."""
    text: str
    confidence: float
    processing_time: float
    
    # Quality metrics
    word_count: int = 0
    line_count: int = 0
    avg_word_confidence: float = 0.0
    
    # Detection results
    detected_layout: DocumentLayout = DocumentLayout.MIXED
    detected_tables: int = 0
    detected_languages: List[str] = field(default_factory=list)
    
    # Processing details
    preprocessing_applied: List[str] = field(default_factory=list)
    ocr_engine_used: str = "tesseract"
    fallback_used: bool = False
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.text:
            words = self.text.split()
            self.word_count = len(words)
            self.line_count = len(self.text.splitlines())


@dataclass
class OCRConfig:
    """Advanced OCR configuration."""
    # Quality settings
    quality_level: OCRQuality = OCRQuality.BALANCED
    confidence_threshold: float = 0.7
    min_word_confidence: int = 30  # Tesseract scale 0-100
    
    # Language settings
    languages: List[str] = field(default_factory=lambda: ['eng'])
    language_detection: bool = True
    
    # Image preprocessing
    dpi: int = 300
    enable_preprocessing: bool = True
    auto_rotate: bool = True
    noise_reduction: bool = True
    contrast_enhancement: bool = True
    
    # Layout detection
    detect_layout: bool = True
    table_detection: bool = True
    form_detection: bool = True
    
    # Performance settings
    timeout_seconds: int = 60
    parallel_processing: bool = True
    
    # Fallback settings
    enable_fallbacks: bool = True
    fallback_quality_levels: List[OCRQuality] = field(
        default_factory=lambda: [OCRQuality.HIGH_QUALITY, OCRQuality.FAST]
    )
    
    # Tesseract specific
    tesseract_configs: Dict[OCRQuality, str] = field(default_factory=lambda: {
        OCRQuality.FAST: '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;-()[]{}"\' ',
        OCRQuality.BALANCED: '--oem 3 --psm 6',
        OCRQuality.HIGH_QUALITY: '--oem 1 --psm 6',
        OCRQuality.ADAPTIVE: '--oem 3 --psm 6'
    })


class OCRProcessor:
    """
    Advanced OCR processor with hybrid approaches and intelligent fallbacks.
    
    Features:
    - Multiple quality levels with automatic fallbacks
    - Document layout detection and optimization
    - Specialized handling for tables, forms, and mixed content
    - Confidence-based quality assessment
    - Image preprocessing pipeline
    - Multi-language support with detection
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """Initialize OCR processor."""
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(__name__)
        
        # Validate Tesseract installation
        self._validate_tesseract()
        
        # Performance tracking
        self.stats = {
            'documents_processed': 0,
            'pages_processed': 0,
            'fallbacks_used': 0,
            'avg_confidence': 0.0,
            'processing_times': []
        }
    
    def _validate_tesseract(self) -> None:
        """Validate Tesseract installation and configuration."""
        try:
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract OCR version: {version}")
            
            # Test basic functionality
            test_image = Image.new('RGB', (100, 50), color='white')
            pytesseract.image_to_string(test_image)
            
        except Exception as e:
            raise OCRProcessingError(
                message=f"Tesseract OCR not available or misconfigured: {str(e)}",
                details={"error": str(e)}
            )
    
    async def process_image(self, image: Union[Image.Image, np.ndarray, str, Path]) -> OCRResult:
        """
        Process image with OCR using hybrid approach and fallbacks.
        
        Args:
            image: PIL Image, numpy array, or path to image file
            
        Returns:
            OCRResult with extracted text and metadata
        """
        start_time = time.time()
        
        try:
            # Load and validate image
            pil_image = self._load_image(image)
            
            # Detect document layout if enabled
            layout = DocumentLayout.MIXED
            if self.config.detect_layout:
                layout = self._detect_layout(pil_image)
            
            # Process with primary quality level
            result = await self._process_with_quality_level(
                pil_image, self.config.quality_level, layout
            )
            
            # Apply fallbacks if confidence is low
            if (result.confidence < self.config.confidence_threshold and 
                self.config.enable_fallbacks):
                
                result = await self._apply_fallbacks(pil_image, result, layout)
            
            # Update statistics
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            if isinstance(e, OCRProcessingError):
                raise
            
            raise OCRProcessingError(
                message=f"OCR processing failed: {str(e)}",
                cause=e
            )
    
    def _load_image(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
        """Load and validate image from various input types."""
        if isinstance(image, Image.Image):
            return image
        
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                return Image.fromarray(image)
        
        elif isinstance(image, (str, Path)):
            # Load from file path
            image_path = Path(image)
            if not image_path.exists():
                raise OCRProcessingError(
                    message=f"Image file not found: {image_path}",
                    details={"image_path": str(image_path)}
                )
            return Image.open(image_path)
        
        else:
            raise ValidationError(
                message=f"Unsupported image type: {type(image)}",
                field_name="image",
                field_value=str(type(image))
            )
    
    def _detect_layout(self, image: Image.Image) -> DocumentLayout:
        """Detect document layout type for optimization."""
        try:
            # Convert to OpenCV format for analysis
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal and vertical lines
            horizontal_lines = self._detect_lines(gray, horizontal=True)
            vertical_lines = self._detect_lines(gray, vertical=True)
            
            # Table detection
            if len(horizontal_lines) > 3 and len(vertical_lines) > 2:
                return DocumentLayout.TABLE
            
            # Form detection (look for checkboxes, form fields)
            if self._detect_form_elements(gray):
                return DocumentLayout.FORM
            
            # Multi-column detection
            if self._detect_columns(gray) > 1:
                return DocumentLayout.MULTI_COLUMN
            
            return DocumentLayout.SINGLE_COLUMN
            
        except Exception as e:
            self.logger.debug(f"Layout detection failed: {e}")
            return DocumentLayout.MIXED
    
    def _detect_lines(self, gray_image: np.ndarray, horizontal: bool = True) -> List[Tuple]:
        """Detect horizontal or vertical lines in image."""
        try:
            # Create line detection kernel
            if horizontal:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Apply morphological operations
            detected_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            lines = []
            min_length = 50 if horizontal else 30
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if (horizontal and w > min_length) or (not horizontal and h > min_length):
                    lines.append((x, y, w, h))
            
            return lines
            
        except Exception as e:
            self.logger.debug(f"Line detection failed: {e}")
            return []
    
    def _detect_form_elements(self, gray_image: np.ndarray) -> bool:
        """Detect form elements like checkboxes and input fields."""
        try:
            # Look for rectangular regions that might be form fields
            contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            form_elements = 0
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check for checkbox-like squares
                if 10 < w < 30 and 10 < h < 30 and abs(w - h) < 5:
                    form_elements += 1
                
                # Check for input field-like rectangles
                elif w > 100 and 15 < h < 40:
                    form_elements += 1
            
            return form_elements > 2
            
        except Exception as e:
            self.logger.debug(f"Form detection failed: {e}")
            return False
    
    def _detect_columns(self, gray_image: np.ndarray) -> int:
        """Detect number of columns in document."""
        try:
            # Simple column detection based on vertical white space
            height, width = gray_image.shape
            
            # Calculate vertical projection (sum of pixel intensities per column)
            vertical_projection = np.sum(gray_image, axis=0)
            
            # Find valleys in the projection (potential column separators)
            mean_intensity = np.mean(vertical_projection)
            valleys = []
            
            for i in range(1, len(vertical_projection) - 1):
                if (vertical_projection[i] > mean_intensity * 1.2 and 
                    vertical_projection[i-1] < vertical_projection[i] and 
                    vertical_projection[i+1] < vertical_projection[i]):
                    valleys.append(i)
            
            # Estimate number of columns
            if len(valleys) == 0:
                return 1
            elif len(valleys) == 1:
                return 2
            else:
                return len(valleys) + 1
                
        except Exception as e:
            self.logger.debug(f"Column detection failed: {e}")
            return 1
    
    async def _process_with_quality_level(
        self, 
        image: Image.Image, 
        quality: OCRQuality, 
        layout: DocumentLayout
    ) -> OCRResult:
        """Process image with specific quality level and layout optimization."""
        try:
            # Preprocess image
            processed_image = image
            preprocessing_applied = []
            
            if self.config.enable_preprocessing:
                processed_image, preprocessing_applied = self._preprocess_image(
                    image, quality, layout
                )
            
            # Configure Tesseract based on layout and quality
            tesseract_config = self._get_tesseract_config(quality, layout)
            
            # Run OCR with confidence data
            ocr_data = pytesseract.image_to_data(
                processed_image,
                config=tesseract_config,
                lang='+'.join(self.config.languages),
                output_type=pytesseract.Output.DICT
            )
            
            # Process OCR results
            result = self._process_ocr_data(ocr_data)
            result.preprocessing_applied = preprocessing_applied
            result.detected_layout = layout
            result.ocr_engine_used = f"tesseract_{quality.value}"
            
            return result
            
        except Exception as e:
            raise OCRProcessingError(
                message=f"OCR processing failed with {quality.value} quality",
                details={
                    "quality_level": quality.value,
                    "layout": layout.value,
                    "error": str(e)
                },
                cause=e
            )
    
    def _preprocess_image(
        self, 
        image: Image.Image, 
        quality: OCRQuality, 
        layout: DocumentLayout
    ) -> Tuple[Image.Image, List[str]]:
        """Preprocess image based on quality level and layout."""
        processed = image.copy()
        applied_steps = []
        
        try:
            # Convert to grayscale if not already
            if processed.mode != 'L':
                processed = processed.convert('L')
                applied_steps.append("grayscale_conversion")
            
            # Auto-rotate if enabled
            if self.config.auto_rotate:
                processed = self._auto_rotate_image(processed)
                applied_steps.append("auto_rotation")
            
            # Noise reduction
            if self.config.noise_reduction and quality in [OCRQuality.BALANCED, OCRQuality.HIGH_QUALITY]:
                processed = processed.filter(ImageFilter.MedianFilter(size=3))
                applied_steps.append("noise_reduction")
            
            # Contrast enhancement
            if self.config.contrast_enhancement:
                enhancer = ImageEnhance.Contrast(processed)
                processed = enhancer.enhance(1.2)
                applied_steps.append("contrast_enhancement")
            
            # Layout-specific preprocessing
            if layout == DocumentLayout.TABLE:
                # Enhance table structure
                processed = self._enhance_table_image(processed)
                applied_steps.append("table_enhancement")
            
            elif layout == DocumentLayout.FORM:
                # Enhance form elements
                processed = self._enhance_form_image(processed)
                applied_steps.append("form_enhancement")
            
            # Quality-specific preprocessing
            if quality == OCRQuality.HIGH_QUALITY:
                # Additional sharpening for high quality
                processed = processed.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
                applied_steps.append("sharpening")
            
            return processed, applied_steps
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image, []
    
    def _auto_rotate_image(self, image: Image.Image) -> Image.Image:
        """Auto-rotate image based on text orientation detection."""
        try:
            # Use Tesseract's orientation detection
            osd = pytesseract.image_to_osd(image)
            
            # Parse rotation angle
            lines = osd.split('\n')
            rotation_angle = 0
            
            for line in lines:
                if 'Rotate:' in line:
                    rotation_angle = int(line.split(':')[1].strip())
                    break
            
            # Rotate if needed
            if rotation_angle != 0:
                return image.rotate(-rotation_angle, expand=True)
            
            return image
            
        except Exception as e:
            self.logger.debug(f"Auto-rotation failed: {e}")
            return image
    
    def _enhance_table_image(self, image: Image.Image) -> Image.Image:
        """Enhance image for better table recognition."""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Enhance table lines
            kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_horizontal)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_vertical)
            
            # Combine enhanced lines with original
            enhanced = cv2.addWeighted(gray, 0.7, horizontal_lines, 0.3, 0)
            enhanced = cv2.addWeighted(enhanced, 0.7, vertical_lines, 0.3, 0)
            
            return Image.fromarray(enhanced)
            
        except Exception as e:
            self.logger.debug(f"Table enhancement failed: {e}")
            return image
    
    def _enhance_form_image(self, image: Image.Image) -> Image.Image:
        """Enhance image for better form recognition."""
        try:
            # Apply slight blur to reduce noise in form fields
            processed = image.filter(ImageFilter.BoxBlur(radius=0.5))
            
            # Enhance contrast for better field recognition
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(1.3)
            
            return processed
            
        except Exception as e:
            self.logger.debug(f"Form enhancement failed: {e}")
            return image
    
    def _get_tesseract_config(self, quality: OCRQuality, layout: DocumentLayout) -> str:
        """Get optimized Tesseract configuration."""
        base_config = self.config.tesseract_configs.get(quality, '--oem 3 --psm 6')
        
        # Layout-specific optimizations
        if layout == DocumentLayout.TABLE:
            base_config += ' --psm 6'  # Uniform block of text
        elif layout == DocumentLayout.FORM:
            base_config += ' --psm 6'  # Uniform block of text
        elif layout == DocumentLayout.SINGLE_COLUMN:
            base_config += ' --psm 3'  # Fully automatic page segmentation
        elif layout == DocumentLayout.MULTI_COLUMN:
            base_config += ' --psm 2'  # Sparse text
        
        return base_config
    
    def _process_ocr_data(self, ocr_data: Dict) -> OCRResult:
        """Process Tesseract OCR data into structured result."""
        try:
            # Extract text with confidence filtering
            words = []
            confidences = []
            
            for i in range(len(ocr_data['text'])):
                word = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])
                
                if word and conf >= self.config.min_word_confidence:
                    words.append(word)
                    confidences.append(conf)
            
            # Join words into text
            text = ' '.join(words)
            
            # Calculate overall confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Detect language if enabled
            detected_languages = []
            if self.config.language_detection and text:
                detected_languages = self._detect_language(text)
            
            return OCRResult(
                text=text,
                confidence=avg_confidence / 100.0,
                processing_time=0.0,  # Will be set by caller
                avg_word_confidence=avg_confidence,
                detected_languages=detected_languages
            )
            
        except Exception as e:
            raise OCRProcessingError(
                message=f"Failed to process OCR data: {str(e)}",
                cause=e
            )
    
    def _detect_language(self, text: str) -> List[str]:
        """Detect language(s) in the text."""
        try:
            # Simple language detection based on character patterns
            # This is a placeholder - you might want to use a proper language detection library
            
            # Check for common English patterns
            if any(word in text.lower() for word in ['the', 'and', 'or', 'to', 'in', 'of']):
                return ['eng']
            
            return ['eng']  # Default to English
            
        except Exception as e:
            self.logger.debug(f"Language detection failed: {e}")
            return ['eng']
    
    async def _apply_fallbacks(
        self, 
        image: Image.Image, 
        initial_result: OCRResult, 
        layout: DocumentLayout
    ) -> OCRResult:
        """Apply fallback strategies if initial result has low confidence."""
        best_result = initial_result
        
        for fallback_quality in self.config.fallback_quality_levels:
            try:
                self.logger.debug(f"Applying fallback with {fallback_quality.value} quality")
                
                fallback_result = await self._process_with_quality_level(
                    image, fallback_quality, layout
                )
                
                # Use fallback if it's better
                if fallback_result.confidence > best_result.confidence:
                    fallback_result.fallback_used = True
                    best_result = fallback_result
                    self.stats['fallbacks_used'] += 1
                    
                    # If we achieved good confidence, stop trying fallbacks
                    if best_result.confidence >= self.config.confidence_threshold:
                        break
                
            except Exception as e:
                self.logger.debug(f"Fallback {fallback_quality.value} failed: {e}")
                continue
        
        return best_result
    
    def _update_stats(self, result: OCRResult) -> None:
        """Update processing statistics."""
        self.stats['pages_processed'] += 1
        self.stats['processing_times'].append(result.processing_time)
        
        # Update average confidence
        current_avg = self.stats['avg_confidence']
        pages_processed = self.stats['pages_processed']
        self.stats['avg_confidence'] = (
            (current_avg * (pages_processed - 1) + result.confidence) / pages_processed
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
            stats['max_processing_time'] = max(stats['processing_times'])
            stats['min_processing_time'] = min(stats['processing_times'])
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of OCR processor."""
        health = {
            "status": "healthy",
            "issues": []
        }
        
        try:
            # Test Tesseract functionality
            test_image = Image.new('RGB', (200, 100), color='white')
            test_result = await self.process_image(test_image)
            
            # Check if basic OCR is working
            if test_result is None:
                health["status"] = "unhealthy"
                health["issues"].append("OCR processing failed on test image")
        
        except Exception as e:
            health["status"] = "unhealthy"
            health["issues"].append(f"OCR health check failed: {e}")
        
        # Add statistics
        health["stats"] = self.get_stats()
        
        return health
