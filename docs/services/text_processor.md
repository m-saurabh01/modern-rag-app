# Text Processor Documentation

## Overview

The `TextProcessor` class provides comprehensive text processing and enhancement capabilities for the Modern RAG application. It implements intelligent text cleaning, normalization, quality assessment, and structure preservation to optimize text content for embedding generation and retrieval accuracy.

## Architecture

The text processor implements a multi-stage processing pipeline:

- **Initial Cleaning**: Unicode normalization, special character handling, OCR artifact removal
- **Language Detection**: Automatic language identification with confidence scoring
- **Text Type Classification**: Domain-specific processing (academic, technical, business, legal, etc.)
- **Quality Assessment**: Comprehensive text quality evaluation with multiple metrics
- **Structure Preservation**: Maintains document hierarchy and formatting for better chunking
- **Enhancement Pipeline**: Applies targeted improvements based on quality analysis

## Class Structure

```python
class TextProcessor:
    def __init__(self, preserve_structure: bool = True)
```

## Methods Documentation

### `__init__(self, preserve_structure: bool = True)`

**Purpose**: Initializes the text processor with configurable structure preservation and NLP resource setup.

**Parameters**:
- `preserve_structure` (bool): Whether to preserve document structure like paragraphs, lists, and headers (default: True)

**What it achieves**:
- Sets up comprehensive text processing capabilities with configurable behavior
- Initializes NLP resources (NLTK, language detection) when available
- Compiles regex patterns for efficient text processing
- Establishes quality assessment thresholds for different text types
- Prepares the processor for multi-language and multi-domain text processing

**Initialization Process**:
1. **NLP Resource Setup**: Validates and downloads required NLTK data
2. **Pattern Compilation**: Compiles regex patterns for efficient text matching
3. **Quality Threshold Configuration**: Sets up quality assessment criteria
4. **Logger Initialization**: Sets up structured logging for processing tracking

**Usage Example**:
```python
# Standard processor with structure preservation
processor = TextProcessor()

# Processor without structure preservation (faster processing)
processor = TextProcessor(preserve_structure=False)
```

### `process_text(self, text: str, text_type: Optional[TextType] = None) -> Dict[str, Any]`

**Purpose**: Main entry point for comprehensive text processing, cleaning, and enhancement.

**Parameters**:
- `text` (str): Raw text content to process
- `text_type` (Optional[TextType]): Optional hint for document type to enable specialized processing

**Returns**:
- `Dict[str, Any]`: Comprehensive processing results containing:
  - `processed_text`: Cleaned and enhanced text content
  - `original_length`: Length of original text in characters
  - `processed_length`: Length of processed text in characters
  - `analysis`: TextAnalysis object with quality metrics and assessment
  - `processing_metadata`: Processing configuration and detection results

**What it achieves**:
- Orchestrates the complete text processing pipeline from raw input to optimized output
- Applies multi-stage cleaning, normalization, and enhancement processes
- Provides comprehensive quality assessment with actionable insights
- Enables specialized processing based on detected or specified document types
- Maintains processing transparency with detailed metadata and analysis

**Processing Pipeline**:
1. **Input Validation**: Handles empty, null, or whitespace-only text
2. **Initial Cleaning**: Unicode normalization, character cleaning, whitespace normalization
3. **Language Detection**: Identifies primary language with confidence scoring
4. **Text Type Detection**: Classifies document type for specialized processing
5. **Advanced Processing**: Applies domain-specific text processing rules
6. **Structure Preservation**: Maintains document hierarchy and formatting
7. **Quality Assessment**: Comprehensive text quality evaluation
8. **Enhancement Application**: Applies targeted improvements based on quality analysis

**Usage Examples**:
```python
# Basic text processing
result = processor.process_text("Your document content here...")
print(f"Quality: {result['analysis'].quality.value}")
print(f"Language: {result['processing_metadata']['language']}")

# Processing with type hint
result = processor.process_text(
    academic_text, 
    text_type=TextType.ACADEMIC
)

# Processing PDF-extracted content
pdf_result = pdf_processor.extract_text_from_pdf("document.pdf")
text_result = processor.process_text(pdf_result['text'])
```

### `_initial_cleaning(self, text: str) -> str`

**Purpose**: Performs initial text cleaning and normalization to prepare text for advanced processing.

**Parameters**:
- `text` (str): Raw text to clean

**Returns**:
- `str`: Initially cleaned text with normalized characters and whitespace

**What it achieves**:
- Normalizes Unicode characters to ensure consistent text representation
- Removes or replaces problematic characters that can interfere with processing
- Standardizes whitespace patterns for consistent document structure
- Removes common OCR artifacts and scanning errors
- Provides foundation for advanced text processing stages

**Cleaning Operations**:
1. **Unicode Normalization**: NFKC normalization for character consistency
2. **Special Character Cleaning**: Standardizes quotes, dashes, ellipses
3. **Whitespace Normalization**: Removes excessive spaces and normalizes line breaks
4. **OCR Artifact Removal**: Cleans common scanning and digitization errors

### `_detect_language(self, text: str) -> Tuple[str, float]`

**Purpose**: Detects the primary language of text content for language-aware processing.

**Parameters**:
- `text` (str): Text to analyze for language detection

**Returns**:
- `Tuple[str, float]`: Language code and confidence score (0.0-1.0)

**What it achieves**:
- Enables language-specific text processing and normalization
- Supports multi-language document processing in RAG systems
- Provides confidence scoring for language detection reliability
- Handles performance optimization by sampling large texts
- Falls back gracefully when language detection libraries are unavailable

**Detection Process**:
1. **Text Sampling**: Uses first 2000 characters for performance on large documents
2. **Language Analysis**: Utilizes langdetect library for statistical language identification
3. **Confidence Scoring**: Provides probability scores for detection reliability
4. **Fallback Handling**: Defaults to English when detection fails or is unavailable

**Supported Languages**: All languages supported by langdetect library (55+ languages)

### `_detect_text_type(self, text: str) -> TextType`

**Purpose**: Classifies text into domain-specific types to enable specialized processing strategies.

**Parameters**:
- `text` (str): Text content to classify

**Returns**:
- `TextType`: Detected document type (ACADEMIC, TECHNICAL, BUSINESS, LEGAL, etc.)

**What it achieves**:
- Enables domain-specific processing optimizations and enhancements
- Improves text quality through specialized cleaning and normalization rules
- Supports targeted chunking strategies based on document characteristics
- Provides context for quality assessment and enhancement decisions
- Facilitates appropriate handling of domain-specific terminology and structure

**Classification Strategy**:
1. **Indicator Analysis**: Counts domain-specific terminology and phrases
2. **Structure Assessment**: Analyzes document formatting and organization patterns
3. **Content Patterns**: Identifies characteristic writing styles and conventions
4. **Threshold Evaluation**: Applies confidence thresholds for classification decisions

**Text Type Categories**:
- **ACADEMIC**: Research papers, theses, academic articles
- **TECHNICAL**: Documentation, specifications, technical manuals
- **BUSINESS**: Reports, proposals, business communications
- **LEGAL**: Contracts, agreements, legal documents
- **STRUCTURED**: Lists, tables, forms, structured data
- **NARRATIVE**: Stories, articles, narrative content
- **GENERAL**: General-purpose text not fitting specific categories

### `_advanced_processing(self, text: str, text_type: TextType, language: str) -> str`

**Purpose**: Applies specialized processing based on detected text type and language characteristics.

**Parameters**:
- `text` (str): Text to process
- `text_type` (TextType): Detected or specified document type
- `language` (str): Detected language code

**Returns**:
- `str`: Text processed with domain and language-specific enhancements

**What it achieves**:
- Implements domain-specific processing rules for optimal text quality
- Applies language-specific normalization and cleaning strategies
- Preserves important domain-specific formatting and terminology
- Optimizes text structure for downstream embedding and retrieval processes
- Maintains document integrity while enhancing processability

**Processing Strategies by Type**:

#### Academic Text Processing (`TextType.ACADEMIC`)
- **Citation Preservation**: Maintains academic citations and reference formatting
- **Section Header Normalization**: Standardizes academic section structure
- **Figure Reference Handling**: Preserves references to figures, tables, and equations
- **Terminology Preservation**: Maintains academic and scientific terminology

#### Technical Text Processing (`TextType.TECHNICAL`)
- **Code Snippet Preservation**: Protects code examples and technical syntax
- **Technical Formatting**: Maintains API references, configuration examples
- **Terminology Standardization**: Normalizes technical terms and abbreviations
- **Structure Enhancement**: Improves technical documentation organization

#### Business Text Processing (`TextType.BUSINESS`)
- **Financial Data Handling**: Preserves monetary values and financial metrics
- **Business Terminology**: Standardizes business and corporate language
- **Report Structure**: Maintains business document formatting conventions
- **Metric Preservation**: Protects key performance indicators and statistics

#### Legal Text Processing (`TextType.LEGAL`)
- **Legal Structure Preservation**: Maintains clause numbering and legal hierarchy
- **Terminology Standardization**: Preserves legal terms and definitions
- **Citation Handling**: Maintains legal citations and references
- **Formatting Preservation**: Protects legal document structural elements

### `_preserve_structure(self, text: str) -> str`

**Purpose**: Preserves document structure and formatting to maintain semantic coherence and improve chunking quality.

**Parameters**:
- `text` (str): Text with structure to preserve

**Returns**:
- `str`: Text with enhanced and preserved structural elements

**What it achieves**:
- Maintains document hierarchy for better semantic understanding
- Preserves paragraph breaks, headers, and list structures
- Enhances chunking effectiveness by maintaining logical boundaries
- Supports better embedding generation through preserved context
- Enables more accurate retrieval by maintaining document organization

**Structure Preservation Elements**:
1. **Paragraph Preservation**: Maintains paragraph breaks and logical text groupings
2. **Header Preservation**: Protects section headers and document hierarchy
3. **List Preservation**: Maintains numbered and bulleted list structures
4. **Enumeration Handling**: Preserves structured enumerations and sequences

### `_analyze_text_quality(self, text: str, language: str, text_type: TextType) -> TextAnalysis`

**Purpose**: Performs comprehensive text quality assessment across multiple dimensions.

**Parameters**:
- `text` (str): Text to analyze
- `language` (str): Detected language for language-aware analysis
- `text_type` (TextType): Text type for context-appropriate assessment

**Returns**:
- `TextAnalysis`: Comprehensive quality analysis with metrics and recommendations

**What it achieves**:
- Provides multi-dimensional quality assessment for informed processing decisions
- Identifies specific quality issues requiring targeted improvements
- Enables quality-based processing strategies and enhancement selection
- Supports quality monitoring and optimization across document collections
- Facilitates automated quality control in document processing pipelines

**Quality Assessment Dimensions**:

#### Basic Metrics
- **Word Count**: Total words for content volume assessment
- **Sentence Count**: Number of sentences for structure analysis
- **Paragraph Count**: Paragraph distribution for organization assessment
- **Average Sentence Length**: Readability and complexity indicator

#### Advanced Metrics
- **Lexical Diversity**: Vocabulary richness (unique words / total words)
- **Readability Score**: Text complexity and accessibility assessment
- **Structure Score**: Document organization and formatting quality
- **Completeness Score**: Content adequacy for document type

#### Quality Classification
- **EXCELLENT (85-100%)**: High-quality, well-structured, comprehensive content
- **GOOD (70-84%)**: Quality content with minor issues
- **FAIR (50-69%)**: Acceptable content with noticeable quality issues
- **POOR (30-49%)**: Problematic content requiring significant enhancement
- **UNUSABLE (0-29%)**: Severely compromised content requiring major intervention

### `_apply_enhancements(self, text: str, analysis: TextAnalysis) -> Tuple[str, List[str]]`

**Purpose**: Applies targeted text enhancements based on quality analysis results.

**Parameters**:
- `text` (str): Text to enhance
- `analysis` (TextAnalysis): Quality analysis results guiding enhancement decisions

**Returns**:
- `Tuple[str, List[str]]`: Enhanced text and list of applied enhancement types

**What it achieves**:
- Implements intelligent enhancement strategies based on identified quality issues
- Applies targeted improvements to address specific text quality problems
- Maintains enhancement transparency through detailed logging of applied changes
- Optimizes text quality for improved embedding generation and retrieval accuracy
- Provides adaptive enhancement strategies based on text type and quality assessment

**Enhancement Categories**:

#### Readability Improvements
- **Sentence Length Optimization**: Breaks overly long sentences for better comprehension
- **Vocabulary Enhancement**: Improves word choice and clarity
- **Flow Enhancement**: Improves text flow and logical progression

#### Structure Improvements  
- **Paragraph Organization**: Adds appropriate paragraph breaks and organization
- **Section Enhancement**: Improves section structure and hierarchy
- **List Formatting**: Enhances list structure and presentation

#### Completeness Improvements
- **Content Gap Filling**: Addresses missing context or information gaps
- **Transition Enhancement**: Improves connections between ideas and sections
- **Context Preservation**: Maintains important contextual information

#### Type-Specific Enhancements
- **Technical Enhancement**: Improves technical documentation clarity and accuracy
- **Academic Enhancement**: Enhances academic writing structure and flow
- **Business Enhancement**: Improves business document professionalism and clarity

## Quality Assessment Framework

### Quality Thresholds and Scoring

The text processor uses configurable quality thresholds for assessment:

```python
quality_thresholds = {
    'min_word_count': 10,           # Minimum words for meaningful content
    'min_sentence_count': 2,        # Minimum sentences for structure
    'max_repetition_ratio': 0.3,    # Maximum acceptable repetition
    'min_lexical_diversity': 0.3,   # Minimum vocabulary diversity
    'min_readability_score': 30.0   # Minimum readability threshold
}
```

### Text Analysis Object

The `TextAnalysis` dataclass provides comprehensive quality metrics:

```python
@dataclass
class TextAnalysis:
    language: str                    # Detected language code
    language_confidence: float       # Detection confidence (0.0-1.0)
    quality: TextQuality            # Overall quality classification
    text_type: TextType             # Document type classification
    readability_score: float        # Readability assessment (0-100)
    structure_score: float          # Structure quality (0.0-1.0)
    completeness_score: float       # Content completeness (0.0-1.0)
    word_count: int                 # Total word count
    sentence_count: int             # Total sentence count
    paragraph_count: int            # Total paragraph count
    avg_sentence_length: float      # Average words per sentence
    lexical_diversity: float        # Vocabulary diversity (0.0-1.0)
    issues: List[str]              # Identified quality issues
    enhancements_applied: List[str] # Applied enhancements
```

## Integration Points

### PDF Processor Integration

```python
# Process PDF-extracted content with text enhancement
pdf_result = pdf_processor.extract_text_from_pdf("document.pdf")
text_result = text_processor.process_text(pdf_result['text'])

# Quality-aware processing pipeline
if text_result['analysis'].quality == TextQuality.POOR:
    # Apply additional processing or flagging
    logger.warning(f"Poor quality text detected: {text_result['analysis'].issues}")
```

### OCR Processor Integration

```python
# Process OCR results with artifact cleaning
ocr_result = ocr_processor.extract_text_from_image(image_data)
text_result = text_processor.process_text(ocr_result['text'])

# OCR-specific enhancements
if ocr_result['confidence'] < 70:
    # Enhanced processing for low-confidence OCR
    text_result = text_processor.process_text(
        ocr_result['text'], 
        text_type=TextType.GENERAL  # Force general processing for uncertain content
    )
```

### Chunking Service Integration

```python
# Process text before chunking for optimal results
text_result = text_processor.process_text(raw_text)

# Use processed text for chunking
chunks = chunking_service.chunk_text(
    text_result['processed_text'],
    preserve_structure=text_processor.preserve_structure
)

# Quality-aware chunking strategy
if text_result['analysis'].quality == TextQuality.EXCELLENT:
    # Use larger chunks for high-quality content
    chunks = chunking_service.chunk_text(text_result['processed_text'], chunk_size=1500)
else:
    # Use smaller chunks for lower quality content
    chunks = chunking_service.chunk_text(text_result['processed_text'], chunk_size=800)
```

### Embedding Service Integration

```python
# Process text before embedding generation
text_result = text_processor.process_text(document_text)

# Generate embeddings from processed text
embeddings = embedding_service.generate_embeddings([text_result['processed_text']])

# Quality-aware embedding strategy
embedding_metadata = {
    'text_quality': text_result['analysis'].quality.value,
    'text_type': text_result['analysis'].text_type.value,
    'language': text_result['processing_metadata']['language'],
    'enhancements': text_result['analysis'].enhancements_applied
}
```

## Usage Examples

### Basic Text Processing

```python
processor = TextProcessor()

# Process simple text
result = processor.process_text("Your document content here...")
print(f"Quality: {result['analysis'].quality.value}")
print(f"Word count: {result['analysis'].word_count}")
print(f"Enhancements: {result['analysis'].enhancements_applied}")
```

### Academic Document Processing

```python
# Process academic paper
academic_text = """
Abstract

This research investigates the methodology and results of machine learning
applications in natural language processing. The analysis demonstrates
significant improvements in accuracy and efficiency.

Introduction

Previous studies have shown that traditional approaches have limitations...
"""

result = processor.process_text(academic_text, text_type=TextType.ACADEMIC)
print(f"Detected type: {result['processing_metadata']['text_type']}")
print(f"Structure score: {result['analysis'].structure_score}")
```

### Technical Documentation Processing

```python
# Process technical documentation
tech_text = """
API Documentation

The REST API provides endpoints for data retrieval and manipulation.
Authentication requires API key in the header.

GET /api/v1/documents
Returns list of available documents with metadata.

POST /api/v1/documents
Creates new document with provided content.
"""

result = processor.process_text(tech_text)
print(f"Readability: {result['analysis'].readability_score}")
print(f"Completeness: {result['analysis'].completeness_score}")
```

### Batch Processing

```python
documents = [
    "Academic research paper content...",
    "Business report with financial data...",
    "Technical specification document...",
    "Legal contract terms and conditions..."
]

processor = TextProcessor()
results = []

for doc in documents:
    result = processor.process_text(doc)
    results.append({
        'text': result['processed_text'],
        'quality': result['analysis'].quality.value,
        'type': result['processing_metadata']['text_type'],
        'word_count': result['analysis'].word_count
    })

# Analysis of batch processing results
quality_distribution = {}
for result in results:
    quality = result['quality']
    quality_distribution[quality] = quality_distribution.get(quality, 0) + 1

print(f"Quality distribution: {quality_distribution}")
```

### Quality-Based Processing Pipeline

```python
def process_document_with_quality_control(text: str) -> Dict[str, Any]:
    """Process document with quality-based enhancement strategy."""
    processor = TextProcessor()
    
    # Initial processing
    result = processor.process_text(text)
    
    # Quality-based decision making
    if result['analysis'].quality == TextQuality.POOR:
        logger.warning("Poor quality text detected, applying aggressive enhancement")
        # Could trigger additional processing or manual review
        
    elif result['analysis'].quality == TextQuality.EXCELLENT:
        logger.info("High quality text, proceeding with standard processing")
        
    # Additional processing based on text type
    if result['analysis'].text_type == TextType.TECHNICAL:
        # Technical documents might need special handling
        logger.info("Technical document detected, preserving code snippets")
    
    return result

# Usage
processed_doc = process_document_with_quality_control(raw_document_text)
```

### Multi-Language Support

```python
# Process documents in different languages
multilingual_docs = {
    'english': "This is an English document about technology...",
    'spanish': "Este es un documento en español sobre tecnología...",
    'french': "Ceci est un document français sur la technologie..."
}

processor = TextProcessor()

for lang_name, text in multilingual_docs.items():
    result = processor.process_text(text)
    detected_lang = result['processing_metadata']['language']
    confidence = result['processing_metadata']['language_confidence']
    
    print(f"{lang_name}: Detected {detected_lang} (confidence: {confidence:.2f})")
```

## Performance Considerations

### Memory Management

- **Efficient Processing**: Processes text in single pass without excessive memory allocation
- **Pattern Compilation**: Pre-compiles regex patterns for performance optimization
- **Sampling Strategy**: Uses text sampling for language detection on large documents
- **Resource Cleanup**: Properly manages NLP resources and temporary data

### Processing Speed

- **Configurable Structure Preservation**: Can disable structure preservation for faster processing
- **Lazy Loading**: Optional dependencies loaded only when needed
- **Batch Processing**: Supports efficient batch processing of multiple documents
- **Caching Strategy**: Compiled patterns and resources cached for reuse

### Scalability Features

- **Memory-Efficient**: Suitable for processing large document collections
- **Configurable Quality Thresholds**: Tunable for different performance/quality tradeoffs
- **Optional Dependencies**: Graceful degradation when advanced NLP libraries unavailable
- **Processing Metrics**: Built-in performance monitoring and logging

## Configuration and Tuning

### Quality Threshold Customization

```python
# Custom quality thresholds
custom_processor = TextProcessor()
custom_processor.quality_thresholds.update({
    'min_word_count': 25,           # Higher minimum for quality assessment
    'min_lexical_diversity': 0.4,   # Stricter vocabulary diversity requirement
    'min_readability_score': 40.0   # Higher readability standard
})
```

### Enhancement Strategy Configuration

```python
# Configure enhancement aggressiveness
processor = TextProcessor(preserve_structure=True)

# Custom enhancement logic
def custom_enhancement_strategy(text: str, analysis: TextAnalysis) -> str:
    if analysis.quality == TextQuality.POOR:
        # Apply aggressive enhancements
        return processor._improve_readability(
            processor._improve_structure(text)
        )
    return text
```

### Performance Tuning

```python
# High-performance configuration
fast_processor = TextProcessor(preserve_structure=False)

# Disable optional features for speed
import services.text_processor
services.text_processor.NLTK_AVAILABLE = False  # Force fallback methods
services.text_processor.LANGDETECT_AVAILABLE = False  # Disable language detection
```

This text processing service provides comprehensive, production-ready text enhancement capabilities that significantly improve the quality of content for RAG applications, with intelligent quality assessment, multi-language support, and domain-specific processing strategies.
