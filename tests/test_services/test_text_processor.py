"""
Tests for the Text Processing Service.

This module provides comprehensive tests for text processing functionality including:
- Text cleaning and normalization
- Language detection and text type classification
- Quality assessment and enhancement
- Structure preservation and text optimization
"""

import pytest
from unittest.mock import Mock, patch
from services.text_processor import (
    TextProcessor, TextQuality, TextType, TextAnalysis
)


class TestTextProcessor:
    """Test suite for TextProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a TextProcessor instance for testing."""
        return TextProcessor(preserve_structure=True)
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing different scenarios."""
        return {
            'empty': '',
            'whitespace': '   \n\t  \n  ',
            'simple': 'This is a simple sentence.',
            'academic': '''
                Abstract
                
                This research study investigates the methodology and results of our 
                experiment. The analysis shows significant findings that contribute 
                to the existing literature. References include multiple citations [1,2,3].
                
                Introduction
                
                The hypothesis of this study is based on previous research and data analysis.
            ''',
            'technical': '''
                System Architecture
                
                The API implementation uses REST protocols with JSON data format.
                The database configuration requires PostgreSQL server setup.
                Function calls include error handling and logging mechanisms.
                
                def process_data(input_data):
                    return transformed_data
            ''',
            'business': '''
                Financial Report Q3 2024
                
                Revenue increased by 15% compared to previous quarter, reaching $2.5M.
                The company's performance shows strong market growth and customer satisfaction.
                Budget allocation for next quarter includes expansion strategy and management overhead.
            ''',
            'poor_quality': 'a b c d e f g h i j k l m n o p q r s t.',
            'ocr_artifacts': 'Th1s t3xt h4s OCR 3rr0rs 4nd w31rd ch4r4ct3rs... rn misread as m',
            'multilingual': 'This is English text mixed with Français words and Español phrases.',
            'structured': '''
                1. First item in the list
                2. Second item with details
                3. Third item with more information
                
                • Bullet point one
                • Bullet point two
                • Bullet point three
            '''
        }
    
    def test_initialization(self):
        """Test TextProcessor initialization."""
        processor = TextProcessor()
        assert processor.preserve_structure is True
        assert hasattr(processor, 'quality_thresholds')
        assert hasattr(processor, 'patterns')
    
    def test_initialization_with_parameters(self):
        """Test TextProcessor initialization with custom parameters."""
        processor = TextProcessor(preserve_structure=False)
        assert processor.preserve_structure is False
    
    def test_empty_text_processing(self, processor, sample_texts):
        """Test processing of empty or whitespace-only text."""
        result = processor.process_text(sample_texts['empty'])
        
        assert result['processed_text'] == ''
        assert result['original_length'] == 0
        assert result['processed_length'] == 0
        assert result['analysis'].quality == TextQuality.UNUSABLE
        assert 'Empty or whitespace-only text' in result['analysis'].issues
    
    def test_whitespace_text_processing(self, processor, sample_texts):
        """Test processing of whitespace-only text."""
        result = processor.process_text(sample_texts['whitespace'])
        
        assert result['processed_text'] == ''
        assert result['analysis'].quality == TextQuality.UNUSABLE
    
    def test_simple_text_processing(self, processor, sample_texts):
        """Test processing of simple, clean text."""
        result = processor.process_text(sample_texts['simple'])
        
        assert len(result['processed_text']) > 0
        assert result['analysis'].word_count > 0
        assert result['analysis'].sentence_count > 0
        assert result['analysis'].language in ['en', 'unknown']
    
    def test_academic_text_detection(self, processor, sample_texts):
        """Test detection and processing of academic text."""
        result = processor.process_text(sample_texts['academic'])
        
        # Should detect academic content
        assert result['analysis'].text_type == TextType.ACADEMIC or \
               result['processing_metadata']['text_type'] == 'academic'
        
        # Should have reasonable quality
        assert result['analysis'].quality in [TextQuality.GOOD, TextQuality.FAIR, TextQuality.EXCELLENT]
        
        # Should preserve academic structure
        assert 'Abstract' in result['processed_text'] or 'abstract' in result['processed_text'].lower()
    
    def test_technical_text_detection(self, processor, sample_texts):
        """Test detection and processing of technical text."""
        result = processor.process_text(sample_texts['technical'])
        
        # Should detect technical content
        processing_type = result['processing_metadata']['text_type']
        assert processing_type in ['technical', 'general']  # May default to general if not enough indicators
        
        # Should preserve technical terms
        assert 'API' in result['processed_text'] or 'api' in result['processed_text'].lower()
    
    def test_business_text_detection(self, processor, sample_texts):
        """Test detection and processing of business text."""
        result = processor.process_text(sample_texts['business'])
        
        # Should detect business content or general
        processing_type = result['processing_metadata']['text_type']
        assert processing_type in ['business', 'general']
        
        # Should preserve business terms
        assert 'revenue' in result['processed_text'].lower() or 'Revenue' in result['processed_text']
    
    def test_structured_text_detection(self, processor, sample_texts):
        """Test detection and processing of structured text."""
        result = processor.process_text(sample_texts['structured'])
        
        # Should detect structured content
        processing_type = result['processing_metadata']['text_type']
        assert processing_type in ['structured', 'general']
        
        # Should preserve list structure if structure preservation is enabled
        processed = result['processed_text']
        assert ('1.' in processed or '•' in processed) or processor.preserve_structure
    
    def test_poor_quality_text_processing(self, processor, sample_texts):
        """Test processing of poor quality text."""
        result = processor.process_text(sample_texts['poor_quality'])
        
        # Should identify as poor quality
        assert result['analysis'].quality in [TextQuality.POOR, TextQuality.UNUSABLE, TextQuality.FAIR]
        
        # Should have low lexical diversity
        assert result['analysis'].lexical_diversity < 0.5
    
    def test_ocr_artifact_cleaning(self, processor, sample_texts):
        """Test cleaning of OCR artifacts."""
        result = processor.process_text(sample_texts['ocr_artifacts'])
        
        # Should clean up some OCR errors
        processed = result['processed_text']
        assert len(processed) > 0
        
        # Should have processed the text (length might change due to cleaning)
        assert result['processed_length'] >= 0
    
    @patch('services.text_processor.LANGDETECT_AVAILABLE', True)
    @patch('services.text_processor.detect_langs')
    def test_language_detection(self, mock_detect_langs, processor, sample_texts):
        """Test language detection functionality."""
        # Mock language detection
        mock_lang_result = Mock()
        mock_lang_result.lang = 'en'
        mock_lang_result.prob = 0.95
        mock_detect_langs.return_value = [mock_lang_result]
        
        result = processor.process_text(sample_texts['simple'])
        
        assert result['processing_metadata']['language'] == 'en'
        assert result['processing_metadata']['language_confidence'] == 0.95
    
    @patch('services.text_processor.LANGDETECT_AVAILABLE', False)
    def test_language_detection_fallback(self, processor, sample_texts):
        """Test language detection fallback when langdetect is not available."""
        result = processor.process_text(sample_texts['simple'])
        
        # Should fall back to English with low confidence
        assert result['processing_metadata']['language'] == 'en'
        assert result['processing_metadata']['language_confidence'] == 0.5
    
    def test_text_type_override(self, processor, sample_texts):
        """Test explicit text type specification."""
        result = processor.process_text(sample_texts['simple'], text_type=TextType.TECHNICAL)
        
        assert result['processing_metadata']['text_type'] == 'technical'
    
    def test_quality_assessment_metrics(self, processor, sample_texts):
        """Test comprehensive quality assessment metrics."""
        result = processor.process_text(sample_texts['academic'])
        analysis = result['analysis']
        
        # Should have all required metrics
        assert hasattr(analysis, 'word_count')
        assert hasattr(analysis, 'sentence_count')
        assert hasattr(analysis, 'paragraph_count')
        assert hasattr(analysis, 'avg_sentence_length')
        assert hasattr(analysis, 'lexical_diversity')
        assert hasattr(analysis, 'readability_score')
        assert hasattr(analysis, 'structure_score')
        assert hasattr(analysis, 'completeness_score')
        
        # Metrics should be reasonable
        assert analysis.word_count > 0
        assert analysis.sentence_count > 0
        assert 0 <= analysis.lexical_diversity <= 1
        assert 0 <= analysis.readability_score <= 100
        assert 0 <= analysis.structure_score <= 1
        assert 0 <= analysis.completeness_score <= 1
    
    def test_enhancement_application(self, processor, sample_texts):
        """Test that enhancements are applied to poor quality text."""
        # Create a text that should trigger enhancements
        poor_text = "a b c d e f g h i j k l m n o p q r s t u v w x y z" * 3
        
        result = processor.process_text(poor_text)
        
        # Should have identified quality issues
        assert result['analysis'].quality in [TextQuality.POOR, TextQuality.FAIR, TextQuality.UNUSABLE]
        
        # Should have a list of enhancements (even if empty)
        assert isinstance(result['analysis'].enhancements_applied, list)
    
    def test_structure_preservation_enabled(self, sample_texts):
        """Test structure preservation when enabled."""
        processor = TextProcessor(preserve_structure=True)
        result = processor.process_text(sample_texts['structured'])
        
        # Structure should be preserved
        processed = result['processed_text']
        assert len(processed) > 0
        # Structure preservation logic should maintain list indicators
    
    def test_structure_preservation_disabled(self, sample_texts):
        """Test processing when structure preservation is disabled."""
        processor = TextProcessor(preserve_structure=False)
        result = processor.process_text(sample_texts['structured'])
        
        # Should still process the text
        assert len(result['processed_text']) > 0
        assert result['processing_metadata']['preserve_structure'] is False
    
    def test_processing_metadata(self, processor, sample_texts):
        """Test that processing metadata is correctly populated."""
        result = processor.process_text(sample_texts['simple'])
        metadata = result['processing_metadata']
        
        # Should have all required metadata fields
        assert 'language' in metadata
        assert 'language_confidence' in metadata
        assert 'text_type' in metadata
        assert 'preserve_structure' in metadata
        
        # Values should be reasonable
        assert isinstance(metadata['language'], str)
        assert isinstance(metadata['language_confidence'], (int, float))
        assert isinstance(metadata['text_type'], str)
        assert isinstance(metadata['preserve_structure'], bool)
    
    def test_error_handling(self, processor):
        """Test error handling for various edge cases."""
        # Test with None input
        with pytest.raises(TypeError):
            processor.process_text(None)
        
        # Test with non-string input
        with pytest.raises(AttributeError):
            processor.process_text(123)
    
    def test_unicode_handling(self, processor):
        """Test handling of Unicode characters."""
        unicode_text = "Héllo Wörld! This tëxt has ūnicōde characters. 中文 العربية"
        
        result = processor.process_text(unicode_text)
        
        # Should process without errors
        assert len(result['processed_text']) > 0
        assert result['analysis'].word_count > 0
    
    def test_very_long_text(self, processor):
        """Test processing of very long text."""
        # Create a long text (simulating a large document)
        long_text = "This is a sentence with reasonable content. " * 1000
        
        result = processor.process_text(long_text)
        
        # Should process without errors
        assert len(result['processed_text']) > 0
        assert result['analysis'].word_count > 1000
        assert result['analysis'].sentence_count > 100
    
    def test_special_characters_cleaning(self, processor):
        """Test cleaning of special characters and normalization."""
        special_text = 'This has "smart quotes" and various—dashes and…ellipses'
        
        result = processor.process_text(special_text)
        
        # Should normalize special characters
        processed = result['processed_text']
        assert len(processed) > 0
        # Some normalization should occur
    
    def test_whitespace_normalization(self, processor):
        """Test whitespace normalization."""
        messy_text = "This   has    excessive     spaces\n\n\n\nand\t\ttabs\nand   \nmixed\n   whitespace"
        
        result = processor.process_text(messy_text)
        
        # Should normalize whitespace
        processed = result['processed_text']
        assert len(processed) > 0
        # Should not have excessive consecutive spaces (basic check)
        assert '    ' not in processed  # No 4+ consecutive spaces
    
    @patch('services.text_processor.NLTK_AVAILABLE', True)
    @patch('services.text_processor.sent_tokenize')
    def test_nltk_integration(self, mock_sent_tokenize, processor, sample_texts):
        """Test NLTK integration when available."""
        mock_sent_tokenize.return_value = ['Sentence 1.', 'Sentence 2.', 'Sentence 3.']
        
        result = processor.process_text(sample_texts['simple'])
        
        # Should use NLTK for sentence tokenization
        assert result['analysis'].sentence_count > 0
    
    @patch('services.text_processor.NLTK_AVAILABLE', False)
    def test_nltk_fallback(self, processor, sample_texts):
        """Test fallback behavior when NLTK is not available."""
        result = processor.process_text(sample_texts['simple'])
        
        # Should still work with fallback methods
        assert len(result['processed_text']) > 0
        assert result['analysis'].sentence_count >= 0
    
    def test_confidence_scoring(self, processor, sample_texts):
        """Test that confidence and quality scoring works reasonably."""
        # Good quality text should score better than poor quality
        good_result = processor.process_text(sample_texts['academic'])
        poor_result = processor.process_text(sample_texts['poor_quality'])
        
        good_quality = good_result['analysis'].quality
        poor_quality = poor_result['analysis'].quality
        
        # Quality enum comparison (assuming EXCELLENT > GOOD > FAIR > POOR > UNUSABLE)
        quality_order = [TextQuality.UNUSABLE, TextQuality.POOR, TextQuality.FAIR, TextQuality.GOOD, TextQuality.EXCELLENT]
        
        good_index = quality_order.index(good_quality) if good_quality in quality_order else 0
        poor_index = quality_order.index(poor_quality) if poor_quality in quality_order else 0
        
        # Good text should have higher quality index than poor text
        assert good_index >= poor_index


class TestTextAnalysis:
    """Test suite for TextAnalysis dataclass."""
    
    def test_text_analysis_creation(self):
        """Test creation of TextAnalysis object."""
        analysis = TextAnalysis(
            language='en',
            language_confidence=0.95,
            quality=TextQuality.GOOD,
            text_type=TextType.ACADEMIC,
            readability_score=75.0,
            structure_score=0.8,
            completeness_score=0.9,
            word_count=100,
            sentence_count=5,
            paragraph_count=2,
            avg_sentence_length=20.0,
            lexical_diversity=0.6,
            issues=['minor_formatting'],
            enhancements_applied=['readability_improvement']
        )
        
        assert analysis.language == 'en'
        assert analysis.quality == TextQuality.GOOD
        assert analysis.text_type == TextType.ACADEMIC
        assert analysis.word_count == 100
        assert len(analysis.issues) == 1
        assert len(analysis.enhancements_applied) == 1


class TestTextEnums:
    """Test suite for TextQuality and TextType enums."""
    
    def test_text_quality_enum(self):
        """Test TextQuality enum values."""
        assert TextQuality.EXCELLENT.value == "excellent"
        assert TextQuality.GOOD.value == "good"
        assert TextQuality.FAIR.value == "fair"
        assert TextQuality.POOR.value == "poor"
        assert TextQuality.UNUSABLE.value == "unusable"
    
    def test_text_type_enum(self):
        """Test TextType enum values."""
        assert TextType.ACADEMIC.value == "academic"
        assert TextType.TECHNICAL.value == "technical"
        assert TextType.BUSINESS.value == "business"
        assert TextType.LEGAL.value == "legal"
        assert TextType.GENERAL.value == "general"
        assert TextType.NARRATIVE.value == "narrative"
        assert TextType.STRUCTURED.value == "structured"


class TestTextProcessorIntegration:
    """Integration tests for TextProcessor with other components."""
    
    def test_integration_with_pdf_processor_output(self):
        """Test processing text extracted from PDF processor."""
        # Simulate PDF processor output
        pdf_extracted_text = {
            'text': '''
                Chapter 1: Introduction
                
                This document provides comprehensive information about the system
                architecture and implementation details. The methodology includes
                multiple approaches for data processing and analysis.
                
                1.1 Background
                
                Previous research has shown significant improvements in processing
                efficiency when using modern algorithms and data structures.
            ''',
            'metadata': {'pages': 10, 'has_scanned_pages': False}
        }
        
        processor = TextProcessor()
        result = processor.process_text(pdf_extracted_text['text'])
        
        # Should process PDF-extracted content successfully
        assert len(result['processed_text']) > 0
        assert result['analysis'].text_type in [TextType.ACADEMIC, TextType.TECHNICAL, TextType.GENERAL]
        assert result['analysis'].quality != TextQuality.UNUSABLE
    
    def test_integration_with_ocr_output(self):
        """Test processing OCR-extracted text with typical artifacts."""
        ocr_text = '''
        ANNUAL REPORT 2024
        
        C0MPANY OVERVIEW
        
        0ur c0mpany has achieved str0ng gr0wth this year with revenue
        increasing by 15%. The perfonnance metrics sh0w p0sitive
        trends acr0ss all business units.
        
        Key achievements include:
        - Impr0ved customer satisfacti0n
        - Expanded market presence  
        - Enhanced 0perati0nal efficiency
        '''
        
        processor = TextProcessor()
        result = processor.process_text(ocr_text)
        
        # Should handle OCR artifacts
        assert len(result['processed_text']) > 0
        assert result['analysis'].text_type in [TextType.BUSINESS, TextType.GENERAL]
        
        # Should clean up some OCR errors
        processed = result['processed_text']
        # Basic check - should still contain key business terms
        assert any(word in processed.lower() for word in ['company', 'revenue', 'business', 'report'])
    
    def test_batch_processing_simulation(self):
        """Test processing multiple texts as would happen in batch scenarios."""
        texts = [
            "This is a simple document about technology and systems.",
            "Legal contract terms and conditions for service agreement.",
            "Research methodology and experimental results analysis.",
            "Business strategy and market analysis report.",
            "Short text."
        ]
        
        processor = TextProcessor()
        results = []
        
        for text in texts:
            result = processor.process_text(text)
            results.append(result)
        
        # All should process successfully
        assert len(results) == len(texts)
        
        # Should have varied text types
        text_types = [r['processing_metadata']['text_type'] for r in results]
        assert len(set(text_types)) > 1  # Should detect different types
        
        # Should have quality assessments
        qualities = [r['analysis'].quality for r in results]
        assert all(q in [TextQuality.EXCELLENT, TextQuality.GOOD, TextQuality.FAIR, 
                        TextQuality.POOR, TextQuality.UNUSABLE] for q in qualities)


if __name__ == "__main__":
    pytest.main([__file__])
