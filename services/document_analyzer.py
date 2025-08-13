"""
Document Structure Analyzer for the Modern RAG Application.

This module provides comprehensive document structure analysis including:
- Document hierarchy detection and mapping
- Table structure identification and preservation
- Section and content block classification
- Entity extraction for government and technical documents
- Structure-aware metadata generation for enhanced chunking
"""

import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging
import time

# Enhanced NLP libraries (all offline-capable)
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.tree import Tree
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Advanced NLP features will be disabled.")

try:
    import spacy
    # Load small English model (works offline once downloaded)
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        # Fallback if model not installed
        SPACY_AVAILABLE = False
        logging.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Advanced entity recognition will be disabled.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas not available. Enhanced table processing will use basic methods.")

# Note: langdetect removed as you mentioned 99% English content


class SectionType(Enum):
    """Document section classification types."""
    HEADER = "header"
    SUBHEADER = "subheader"
    TITLE = "title"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CAPTION = "caption"
    REFERENCE = "reference"
    METADATA = "metadata"
    FOOTER = "footer"


class EntityType(Enum):
    """Entity classification types for document analysis."""
    # Government entities
    DEPARTMENT = "department"
    REFERENCE_NUMBER = "reference_number"
    DOCUMENT_DATE = "document_date"
    OFFICIAL_TITLE = "official_title"
    
    # Technical entities
    SYSTEM_NAME = "system_name"
    VERSION_NUMBER = "version_number"
    REQUIREMENT_ID = "requirement_id"
    COMPONENT = "component"
    
    # General entities
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    PHONE = "phone"
    EMAIL = "email"


@dataclass
class DocumentSection:
    """Represents a document section with hierarchy and metadata."""
    section_id: str
    section_type: SectionType
    title: str
    content: str
    level: int  # Hierarchy level (0=top level, 1=subsection, etc.)
    start_position: int
    end_position: int
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []


@dataclass
class TableStructure:
    """Represents table structure and metadata."""
    table_id: str
    caption: str
    headers: List[str]
    rows: List[List[str]]
    start_position: int
    end_position: int
    table_type: str = "data"  # data, financial, schedule, list
    has_headers: bool = True
    column_count: int = 0
    row_count: int = 0
    
    def __post_init__(self):
        if not self.column_count and self.headers:
            self.column_count = len(self.headers)
        if not self.row_count:
            self.row_count = len(self.rows)


@dataclass
class DocumentEntity:
    """Represents extracted entities with context."""
    entity_id: str
    entity_type: EntityType
    text: str
    context: str
    confidence: float
    start_position: int
    end_position: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentStructure:
    """Complete document structure analysis results."""
    document_id: str
    document_type: str
    title: str
    sections: List[DocumentSection]
    tables: List[TableStructure]
    entities: List[DocumentEntity]
    hierarchy: Dict[str, List[str]]  # section_id -> children_ids
    metadata: Dict[str, Any]
    processing_time: float = 0.0
    
    def __post_init__(self):
        if not self.hierarchy:
            self.hierarchy = self._build_hierarchy()
    
    def _build_hierarchy(self) -> Dict[str, List[str]]:
        """Build section hierarchy mapping."""
        hierarchy = {}
        for section in self.sections:
            if section.parent_id:
                if section.parent_id not in hierarchy:
                    hierarchy[section.parent_id] = []
                hierarchy[section.parent_id].append(section.section_id)
        return hierarchy


class DocumentAnalyzer:
    """
    Advanced document structure analyzer for complex document processing.
    
    Provides fast, rule-based document analysis with table detection,
    entity extraction, and structure preservation optimized for RAG applications.
    """
    
    def __init__(self):
        """Initialize the document analyzer with optimized patterns and NLP tools."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP tools
        self._setup_nltk()
        self._setup_spacy()
        
        # Compile regex patterns for performance
        self._compile_patterns()
        
        # Entity extraction patterns
        self._compile_entity_patterns()
        
        # Document structure indicators
        self._compile_structure_patterns()
    
    def _setup_nltk(self):
        """Ensure NLTK data is available for offline processing."""
        if NLTK_AVAILABLE:
            try:
                # Check for required data and download if missing
                required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
                for dataset in required_data:
                    try:
                        nltk.data.find(f'tokenizers/{dataset}')
                    except LookupError:
                        try:
                            nltk.data.find(f'corpora/{dataset}')
                        except LookupError:
                            try:
                                nltk.data.find(f'taggers/{dataset}')
                            except LookupError:
                                try:
                                    nltk.data.find(f'chunkers/{dataset}')
                                except LookupError:
                                    self.logger.info(f"Downloading NLTK dataset: {dataset}")
                                    nltk.download(dataset, quiet=True)
                
                self.nltk_ready = True
                self.logger.info("NLTK initialized successfully for offline processing")
            except Exception as e:
                self.logger.warning(f"NLTK setup failed: {e}. Falling back to basic processing.")
                self.nltk_ready = False
        else:
            self.nltk_ready = False
    
    def _setup_spacy(self):
        """Initialize spaCy for offline processing."""
        if SPACY_AVAILABLE:
            try:
                # Verify the model works
                test_doc = nlp("Test sentence.")
                self.spacy_ready = True
                self.logger.info("spaCy initialized successfully for offline processing")
            except Exception as e:
                self.logger.warning(f"spaCy setup failed: {e}. Falling back to basic entity extraction.")
                self.spacy_ready = False
        else:
            self.spacy_ready = False
    
    def analyze_document(self, text: str, document_id: str = "unknown") -> DocumentStructure:
        """
        Main entry point for comprehensive document structure analysis.
        
        Args:
            text: Document text to analyze
            document_id: Unique identifier for the document
            
        Returns:
            DocumentStructure with complete analysis results
        """
        import time
        start_time = time.time()
        
        self.logger.info("Starting document structure analysis", extra={
            'document_id': document_id,
            'text_length': len(text)
        })
        
        try:
            # Step 1: Document type detection
            doc_type = self._detect_document_type(text)
            
            # Step 2: Extract document title
            title = self._extract_document_title(text)
            
            # Step 3: Detect and extract tables
            tables = self._detect_tables(text)
            
            # Step 4: Analyze document sections and hierarchy
            sections = self._analyze_sections(text, tables)
            
            # Step 5: Extract entities
            entities = self._extract_entities(text, doc_type)
            
            # Step 6: Generate document metadata
            metadata = self._generate_metadata(text, doc_type, sections, tables, entities)
            
            processing_time = time.time() - start_time
            
            structure = DocumentStructure(
                document_id=document_id,
                document_type=doc_type,
                title=title,
                sections=sections,
                tables=tables,
                entities=entities,
                hierarchy={},
                metadata=metadata,
                processing_time=processing_time
            )
            
            self.logger.info("Document structure analysis completed", extra={
                'document_id': document_id,
                'processing_time': processing_time,
                'sections_count': len(sections),
                'tables_count': len(tables),
                'entities_count': len(entities)
            })
            
            return structure
            
        except Exception as e:
            self.logger.error("Document structure analysis failed", extra={
                'document_id': document_id,
                'error': str(e),
                'text_length': len(text)
            })
            raise
    
    def _detect_document_type(self, text: str) -> str:
        """
        Fast document type detection based on content patterns.
        
        Args:
            text: Document text to analyze
            
        Returns:
            Detected document type string
        """
        text_lower = text.lower()
        
        # Government document indicators
        gov_indicators = [
            'department of', 'ministry of', 'government of', 'federal',
            'notice', 'circular', 'memorandum', 'directive', 'order',
            'regulation', 'policy', 'act', 'bill', 'amendment'
        ]
        
        # Technical manual indicators
        tech_indicators = [
            'user manual', 'technical guide', 'installation', 'configuration',
            'setup', 'troubleshooting', 'maintenance', 'operation',
            'system requirements', 'specifications'
        ]
        
        # Change request indicators
        cr_indicators = [
            'change request', 'cr-', 'modification', 'enhancement',
            'bug fix', 'feature request', 'development', 'implementation',
            'requirements', 'acceptance criteria'
        ]
        
        # Business document indicators
        business_indicators = [
            'report', 'proposal', 'analysis', 'strategy', 'plan',
            'budget', 'financial', 'quarterly', 'annual', 'summary'
        ]
        
        # Count indicators (fast approach)
        gov_score = sum(1 for indicator in gov_indicators if indicator in text_lower)
        tech_score = sum(1 for indicator in tech_indicators if indicator in text_lower)
        cr_score = sum(1 for indicator in cr_indicators if indicator in text_lower)
        business_score = sum(1 for indicator in business_indicators if indicator in text_lower)
        
        # Determine type based on highest score
        scores = {
            'government': gov_score,
            'technical_manual': tech_score,
            'change_request': cr_score,
            'business_report': business_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        
        return 'general'
    
    def _extract_document_title(self, text: str) -> str:
        """
        Extract document title from various positions and formats.
        
        Args:
            text: Document text
            
        Returns:
            Extracted document title or default
        """
        lines = text.split('\n')[:10]  # Check first 10 lines
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip common headers/footers
            if any(skip in line.lower() for skip in ['page ', 'confidential', 'draft', 'version']):
                continue
            
            # Look for title patterns
            if len(line) > 10 and len(line) < 200:
                # Check if line looks like a title
                if (line.isupper() or 
                    line.istitle() or 
                    re.match(r'^[A-Z][^.!?]*$', line)):
                    return line
        
        # Fallback: use first substantial line
        for line in lines:
            line = line.strip()
            if len(line) > 20:
                return line[:100] + ('...' if len(line) > 100 else '')
        
        return "Untitled Document"
    
    def _detect_tables(self, text: str) -> List[TableStructure]:
        """
        Fast rule-based table detection and extraction.
        
        Args:
            text: Document text containing potential tables
            
        Returns:
            List of detected table structures
        """
        tables = []
        lines = text.split('\n')
        
        i = 0
        table_id = 0
        
        while i < len(lines):
            # Look for table patterns
            table_start = self._find_table_start(lines, i)
            if table_start is not None:
                table_end = self._find_table_end(lines, table_start)
                if table_end is not None and table_end > table_start:
                    # Extract table
                    table_lines = lines[table_start:table_end + 1]
                    table = self._parse_table(table_lines, table_id, table_start)
                    if table:
                        tables.append(table)
                        table_id += 1
                    i = table_end + 1
                else:
                    i += 1
            else:
                i += 1
        
        return tables
    
    def _find_table_start(self, lines: List[str], start_idx: int) -> Optional[int]:
        """Find the start of a table structure."""
        for i in range(start_idx, min(start_idx + 10, len(lines))):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Look for table indicators
            if (self._is_table_separator(line) or 
                self._is_table_row(line) or
                self._has_multiple_columns(line)):
                return i
        
        return None
    
    def _find_table_end(self, lines: List[str], start_idx: int) -> Optional[int]:
        """Find the end of a table structure."""
        consecutive_empty = 0
        last_table_line = start_idx
        
        for i in range(start_idx + 1, min(start_idx + 50, len(lines))):
            line = lines[i].strip()
            
            if not line:
                consecutive_empty += 1
                if consecutive_empty >= 2:  # End table after 2+ empty lines
                    break
            else:
                consecutive_empty = 0
                if (self._is_table_row(line) or 
                    self._is_table_separator(line) or
                    self._has_multiple_columns(line)):
                    last_table_line = i
                else:
                    # Check if this looks like non-table content
                    if not self._could_be_table_content(line):
                        break
        
        return last_table_line if last_table_line > start_idx else None
    
    def _parse_table(self, table_lines: List[str], table_id: int, start_pos: int) -> Optional[TableStructure]:
        """Parse table lines into structured table data with enhanced pandas support."""
        if not table_lines:
            return None
        
        # Try pandas-enhanced parsing first
        if PANDAS_AVAILABLE:
            pandas_table = self._parse_table_with_pandas(table_lines, table_id, start_pos)
            if pandas_table:
                return pandas_table
        
        # Fallback to basic parsing
        return self._parse_table_basic(table_lines, table_id, start_pos)
    
    def _parse_table_with_pandas(self, table_lines: List[str], table_id: int, start_pos: int) -> Optional[TableStructure]:
        """Enhanced table parsing using pandas for better accuracy."""
        try:
            # Clean lines and remove separators
            data_lines = []
            for line in table_lines:
                line = line.strip()
                if line and not self._is_table_separator(line):
                    data_lines.append(line)
            
            if len(data_lines) < 2:
                return None
            
            # Determine delimiter
            if '|' in data_lines[0]:
                # Pipe-separated table
                delimiter = '|'
                # Clean pipe table format
                clean_lines = []
                for line in data_lines:
                    # Remove leading/trailing pipes and split
                    clean_line = line.strip('|').strip()
                    if clean_line:
                        clean_lines.append(clean_line)
                
                # Convert to pandas-readable format
                table_text = '\n'.join(clean_lines)
                
            elif '\t' in data_lines[0]:
                # Tab-separated
                delimiter = '\t'
                table_text = '\n'.join(data_lines)
            else:
                # Space-separated - try to detect consistent spacing
                delimiter = None  # pandas will auto-detect
                table_text = '\n'.join(data_lines)
            
            # Use pandas to parse
            from io import StringIO
            
            if delimiter == '|':
                # For pipe tables, split manually for better control
                rows = []
                for line in clean_lines:
                    cols = [col.strip() for col in line.split('|') if col.strip()]
                    if cols:
                        rows.append(cols)
                
                if len(rows) < 2:
                    return None
                
                headers = rows[0]
                data_rows = rows[1:]
                
            else:
                # Use pandas read_csv for complex parsing
                df = pd.read_csv(StringIO(table_text), delimiter=delimiter, engine='python')
                headers = df.columns.tolist()
                data_rows = df.values.tolist()
            
            if not headers or not data_rows:
                return None
            
            # Classify table type with enhanced logic
            table_type = self._classify_table_type_enhanced(headers, data_rows)
            
            return TableStructure(
                table_id=f"table_{table_id}",
                headers=[str(h).strip() for h in headers],
                rows=[[str(cell).strip() for cell in row] for row in data_rows],
                position=start_pos,
                column_count=len(headers),
                row_count=len(data_rows),
                table_type=table_type
            )
            
        except Exception as e:
            self.logger.debug(f"pandas table parsing failed: {e}, falling back to basic parsing")
            return None
    
    def _parse_table_basic(self, table_lines: List[str], table_id: int, start_pos: int) -> Optional[TableStructure]:
        """Basic table parsing without pandas."""
        # Clean lines and remove separators
        data_lines = []
        for line in table_lines:
            line = line.strip()
            if line and not self._is_table_separator(line):
                data_lines.append(line)

        if len(data_lines) < 2:  # Need at least header + 1 row
            return None

        # Parse table data
        rows = []
        headers = []

        for i, line in enumerate(data_lines):
            columns = self._split_table_columns(line)
            if columns:
                if i == 0:  # First row as headers
                    headers = columns
                else:
                    rows.append(columns)

        if not headers or not rows:
            return None
        
        # Extract caption (look for lines before table)
        caption = f"Table {table_id + 1}"
        
        return TableStructure(
            table_id=f"table_{table_id}",
            caption=caption,
            headers=headers,
            rows=rows,
            start_position=start_pos,
            end_position=start_pos + len(table_lines),
            table_type=self._classify_table_type(headers, rows),
            has_headers=True,
            column_count=len(headers),
            row_count=len(rows)
        )
    
    def _analyze_sections(self, text: str, tables: List[TableStructure]) -> List[DocumentSection]:
        """
        Analyze document sections and create hierarchy with enhanced NLP.
        
        Args:
            text: Document text
            tables: Detected tables to exclude from section analysis
            
        Returns:
            List of document sections with hierarchy
        """
        sections = []
        
        # Use NLTK for better sentence segmentation if available
        if self.nltk_ready:
            sections = self._analyze_sections_nltk(text, tables)
        
        # Fallback to basic line-by-line analysis
        if not sections:
            sections = self._analyze_sections_basic(text, tables)
        
        # Establish hierarchy
        self._establish_section_hierarchy(sections)
        
        return sections
    
    def _analyze_sections_nltk(self, text: str, tables: List[TableStructure]) -> List[DocumentSection]:
        """Enhanced section analysis using NLTK for better text understanding."""
        sections = []
        
        try:
            # Split into sentences for better boundary detection
            sentences = sent_tokenize(text)
            lines = text.split('\n')
            
            # Create table position map for exclusion
            table_positions = set()
            for table in tables:
                start_pos = getattr(table, 'start_position', table.position)
                end_pos = getattr(table, 'end_position', table.position + 1000)
                for pos in range(start_pos, end_pos):
                    table_positions.add(pos)
            
            section_id = 0
            current_position = 0
            
            # Analyze line by line with sentence context
            for i, line in enumerate(lines):
                line_start = current_position
                line_end = current_position + len(line)
                current_position = line_end + 1  # +1 for newline
                
                # Skip if line is within a table
                if any(pos in table_positions for pos in range(line_start, line_end)):
                    continue
                
                section_type = self._classify_line_enhanced(line, sentences)
                if section_type != SectionType.PARAGRAPH or line.strip():
                    level = self._determine_section_level(line, section_type)
                    
                    # Enhanced title extraction using NLP
                    title = self._extract_section_title_enhanced(line, section_type)
                    
                    section = DocumentSection(
                        section_id=f"section_{section_id}",
                        section_type=section_type,
                        title=title,
                        content=line.strip(),
                        level=level,
                        start_position=line_start,
                        end_position=line_end,
                        parent_id=None,  # Will be set in hierarchy establishment
                        children_ids=[]
                    )
                    sections.append(section)
                    section_id += 1
            
            return sections
            
        except Exception as e:
            self.logger.debug(f"NLTK section analysis failed: {e}, falling back to basic")
            return []
    
    def _analyze_sections_basic(self, text: str, tables: List[TableStructure]) -> List[DocumentSection]:
        """Basic section analysis without NLP enhancement."""
        sections = []
        lines = text.split('\n')
        
        # Create table position map for exclusion
        table_positions = set()
        for table in tables:
            start_pos = getattr(table, 'start_position', table.position)
            end_pos = getattr(table, 'end_position', table.position + 1000)
            for pos in range(start_pos, end_pos):
                table_positions.add(pos)
        
        section_id = 0
        current_position = 0
        
        for i, line in enumerate(lines):
            line_start = current_position
            line_end = current_position + len(line)
            current_position = line_end + 1  # +1 for newline
            
            # Skip if line is within a table
            if any(pos in table_positions for pos in range(line_start, line_end)):
                continue
                
            section_type = self._classify_line(line)
            if section_type != SectionType.PARAGRAPH or line.strip():
                level = self._determine_section_level(line, section_type)
                title = line.strip()[:50] + "..." if len(line.strip()) > 50 else line.strip()
                
                section = DocumentSection(
                    section_id=f"section_{section_id}",
                    section_type=section_type,
                    title=title,
                    content=line.strip(),
                    level=level,
                    start_position=line_start,
                    end_position=line_end,
                    parent_id=None,
                    children_ids=[]
                )
                sections.append(section)
                section_id += 1
        
        return sections
    
    def _classify_line_enhanced(self, line: str, sentences: List[str]) -> SectionType:
        """Enhanced line classification using sentence context."""
        # Use basic classification first
        basic_type = self._classify_line(line)
        
        # Enhanced classification with sentence context
        line_text = line.strip()
        
        if not line_text:
            return SectionType.PARAGRAPH
        
        # Check if this line appears to be part of a larger sentence
        if self.nltk_ready:
            try:
                # Check if line ends with sentence-ending punctuation
                ends_sentence = line_text.endswith(('.', '!', '?', ':', ';'))
                
                # Check if line starts with capital letter
                starts_capitalized = line_text[0].isupper() if line_text else False
                
                # If it's a complete sentence and looks like a header, classify as header
                if (basic_type == SectionType.HEADER and ends_sentence and starts_capitalized and
                    len(line_text.split()) <= 10):  # Headers are usually short
                    return SectionType.HEADER
                
                # Check for subheader patterns
                if (line_text.startswith(('1.', '2.', '3.', '4.', '5.', 'A.', 'B.', 'C.')) and
                    len(line_text.split()) <= 15):
                    return SectionType.SUBHEADER
                    
            except Exception:
                pass  # Fall back to basic classification
        
        return basic_type
    
    def _extract_section_title_enhanced(self, line: str, section_type: SectionType) -> str:
        """Enhanced title extraction using NLP patterns."""
        line_text = line.strip()
        
        if section_type in [SectionType.HEADER, SectionType.SUBHEADER]:
            # Remove numbering patterns for cleaner titles
            title = re.sub(r'^[0-9]+\.?\s*', '', line_text)  # Remove "1. "
            title = re.sub(r'^[A-Z]\.?\s*', '', title)       # Remove "A. "
            title = re.sub(r'^[IVX]+\.?\s*', '', title)      # Remove Roman numerals
            
            # Clean up common header artifacts
            title = re.sub(r'^[:\-\=\*]+\s*', '', title)     # Remove leading symbols
            title = title.strip()
            
            # Capitalize properly if it's all caps
            if title.isupper() and len(title) > 3:
                title = title.title()
            
            return title if title else line_text
        
        # For other types, return a reasonable length snippet
        if len(line_text) > 50:
            return line_text[:47] + "..."
        
        return line_text
    
    def _extract_entities(self, text: str, doc_type: str) -> List[DocumentEntity]:
        """
        Extract entities from document text using enhanced NLP when available.
        
        Args:
            text: Document text
            doc_type: Document type for targeted extraction
            
        Returns:
            List of extracted entities
        """
        entities = []
        entity_id = 0
        
        # Enhanced entity extraction with spaCy (if available)
        if self.spacy_ready:
            entities.extend(self._extract_entities_spacy(text, entity_id))
            entity_id += len(entities)
        
        # Enhanced entity extraction with NLTK (if available)
        if self.nltk_ready:
            nltk_entities = self._extract_entities_nltk(text, entity_id)
            # Merge with existing entities (avoid duplicates)
            entities.extend(self._merge_entities(entities, nltk_entities))
            entity_id = len(entities)
        
        # Universal entity extraction (regex-based)
        universal_entities = self._extract_universal_entities(text, entity_id)
        entities.extend(self._merge_entities(entities, universal_entities))
        entity_id = len(entities)
        
        # Type-specific entity extraction
        if doc_type == 'government':
            entities.extend(self._extract_government_entities(text, entity_id))
        elif doc_type == 'technical_manual':
            entities.extend(self._extract_technical_entities(text, entity_id))
        elif doc_type == 'change_request':
            entities.extend(self._extract_cr_entities(text, entity_id))
        
        return entities
    
    def _extract_entities_spacy(self, text: str, start_id: int) -> List[DocumentEntity]:
        """Extract entities using spaCy NER (offline)."""
        entities = []
        entity_id = start_id
        
        try:
            # Process text with spaCy
            doc = nlp(text)
            
            for ent in doc.ents:
                # Map spaCy entity types to our EntityType
                entity_type = self._map_spacy_entity_type(ent.label_)
                if entity_type:
                    context = self._get_context(text, ent.start_char, ent.end_char, 50)
                    
                    entity = DocumentEntity(
                        entity_id=f"entity_{entity_id}",
                        entity_type=entity_type,
                        text=ent.text.strip(),
                        position=ent.start_char,
                        confidence=0.9,  # spaCy entities are generally high confidence
                        context=context,
                        section_id=None  # Will be updated later if needed
                    )
                    entities.append(entity)
                    entity_id += 1
        
        except Exception as e:
            self.logger.debug(f"spaCy entity extraction failed: {e}")
        
        return entities
    
    def _extract_entities_nltk(self, text: str, start_id: int) -> List[DocumentEntity]:
        """Extract entities using NLTK (offline)."""
        entities = []
        entity_id = start_id
        
        try:
            # Tokenize and tag
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                words = word_tokenize(sentence)
                pos_tags = pos_tag(words)
                
                # Named entity chunking
                tree = ne_chunk(pos_tags, binary=False)
                
                current_pos = text.find(sentence)
                word_pos = 0
                
                for subtree in tree:
                    if isinstance(subtree, Tree):
                        # This is a named entity
                        entity_text = " ".join([token for token, pos in subtree.leaves()])
                        entity_label = subtree.label()
                        
                        # Map NLTK entity types to our EntityType
                        entity_type = self._map_nltk_entity_type(entity_label)
                        if entity_type:
                            # Find position in text
                            entity_start = text.find(entity_text, current_pos)
                            if entity_start != -1:
                                context = self._get_context(text, entity_start, 
                                                          entity_start + len(entity_text), 50)
                                
                                entity = DocumentEntity(
                                    entity_id=f"entity_{entity_id}",
                                    entity_type=entity_type,
                                    text=entity_text.strip(),
                                    position=entity_start,
                                    confidence=0.8,  # NLTK entities are good confidence
                                    context=context,
                                    section_id=None
                                )
                                entities.append(entity)
                                entity_id += 1
                    
                    # Update position tracking
                    if isinstance(subtree, Tree):
                        word_pos += len(subtree.leaves())
                    else:
                        word_pos += 1
        
        except Exception as e:
            self.logger.debug(f"NLTK entity extraction failed: {e}")
        
        return entities
    
    def _map_spacy_entity_type(self, spacy_label: str) -> Optional['EntityType']:
        """Map spaCy entity labels to our EntityType enum."""
        spacy_mapping = {
            'PERSON': EntityType.PERSON_NAME,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,
            'MONEY': EntityType.MONETARY_AMOUNT,
            'DATE': EntityType.DATE,
            'TIME': EntityType.DATE,  # We treat time as date
            'WORK_OF_ART': EntityType.REFERENCE_NUMBER,  # Documents, etc.
            'LAW': EntityType.REFERENCE_NUMBER,  # Legal references
        }
        return spacy_mapping.get(spacy_label)
    
    def _map_nltk_entity_type(self, nltk_label: str) -> Optional['EntityType']:
        """Map NLTK entity labels to our EntityType enum."""
        nltk_mapping = {
            'PERSON': EntityType.PERSON_NAME,
            'ORGANIZATION': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,  # Geopolitical entity
            'LOCATION': EntityType.LOCATION,
            'MONEY': EntityType.MONETARY_AMOUNT,
        }
        return nltk_mapping.get(nltk_label)
    
    def _merge_entities(self, existing: List[DocumentEntity], new: List[DocumentEntity]) -> List[DocumentEntity]:
        """Merge entity lists, removing duplicates based on text and position."""
        merged = []
        existing_set = {(e.text.lower(), e.position) for e in existing}
        
        for entity in new:
            key = (entity.text.lower(), entity.position)
            # Allow some position tolerance for same text
            is_duplicate = any(
                abs(entity.position - pos) < 10 and text == entity.text.lower()
                for text, pos in existing_set
            )
            
            if not is_duplicate:
                merged.append(entity)
                existing_set.add(key)
        
        return merged
    
    def _classify_table_type_enhanced(self, headers: List[str], rows: List[List[str]]) -> str:
        """Enhanced table type classification using NLP insights."""
        header_text = ' '.join(headers).lower()
        
        # Financial indicators (enhanced with NLP)
        financial_keywords = ['amount', 'cost', 'price', 'revenue', 'profit', 'budget', 'expense', 
                             'total', 'sum', 'balance', 'payment', 'invoice', 'billing']
        financial_score = sum(1 for keyword in financial_keywords if keyword in header_text)
        
        # Schedule/Time indicators
        schedule_keywords = ['date', 'time', 'deadline', 'schedule', 'start', 'end', 'duration', 
                            'begin', 'finish', 'period', 'phase', 'milestone']
        schedule_score = sum(1 for keyword in schedule_keywords if keyword in header_text)
        
        # Data/Technical indicators
        data_keywords = ['id', 'name', 'type', 'status', 'value', 'parameter', 'setting', 
                        'configuration', 'specification', 'requirement', 'version']
        data_score = sum(1 for keyword in data_keywords if keyword in header_text)
        
        # Determine type based on scores
        if financial_score > 0 and financial_score >= max(schedule_score, data_score):
            return 'financial'
        elif schedule_score > 0 and schedule_score >= max(financial_score, data_score):
            return 'schedule'
        elif data_score > 0:
            return 'data'
        elif len(headers) <= 2:
            return 'list'
        else:
            return 'data'  # Default for structured data
    
    def _generate_metadata(self, text: str, doc_type: str, sections: List[DocumentSection], 
                          tables: List[TableStructure], entities: List[DocumentEntity]) -> Dict[str, Any]:
        """Generate comprehensive document metadata."""
        word_count = len(text.split())
        line_count = len(text.split('\n'))
        
        # Entity counts by type
        entity_counts = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Section type distribution
        section_counts = {}
        for section in sections:
            section_type = section.section_type.value
            section_counts[section_type] = section_counts.get(section_type, 0) + 1
        
        return {
            'document_type': doc_type,
            'word_count': word_count,
            'line_count': line_count,
            'section_count': len(sections),
            'table_count': len(tables),
            'entity_count': len(entities),
            'max_section_level': max([s.level for s in sections]) if sections else 0,
            'has_structured_content': len(tables) > 0 or len([s for s in sections if s.section_type == SectionType.LIST_ITEM]) > 0,
            'entity_distribution': entity_counts,
            'section_distribution': section_counts,
            'complexity_score': self._calculate_complexity_score(word_count, len(sections), len(tables), len(entities))
        }
    
    # Pattern compilation methods
    def _compile_patterns(self):
        """Compile regex patterns for document analysis."""
        self.patterns = {
            # Table patterns
            'table_separator': re.compile(r'^[\s]*[|\-+=\s]{4,}[\s]*$'),
            'table_border': re.compile(r'^[\s]*[\+\-\|]{3,}[\s]*$'),
            'column_separator': re.compile(r'[\|]{1,}|\t{1,}|\s{3,}'),
            
            # Header patterns
            'numbered_header': re.compile(r'^\s*(\d+\.)+\s*.+'),
            'lettered_header': re.compile(r'^\s*[A-Z]\.\s*.+'),
            'section_header': re.compile(r'^\s*[A-Z][A-Z\s]{3,}$'),
            
            # List patterns
            'bullet_list': re.compile(r'^\s*[•\-\*]\s+'),
            'numbered_list': re.compile(r'^\s*\d+\.\s+'),
            
            # Reference patterns
            'reference_line': re.compile(r'^\s*(ref|reference|see|cf\.)', re.IGNORECASE)
        }
    
    def _compile_entity_patterns(self):
        """Compile entity extraction patterns."""
        self.entity_patterns = {
            # Universal patterns
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'[\+]?[1-9]?[0-9]{7,12}'),
            'date': re.compile(r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b|\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b'),
            
            # Government patterns
            'department': re.compile(r'(Department of|Ministry of|Office of|Bureau of)\s+[A-Z][A-Za-z\s&]+', re.IGNORECASE),
            'reference_number': re.compile(r'(Ref|Reference|No|Number)\.?\s*:?\s*([A-Z0-9\-/]+)', re.IGNORECASE),
            'official_title': re.compile(r'(Director|Secretary|Minister|Commissioner|Chief)\s+[A-Z][A-Za-z\s]+', re.IGNORECASE),
            
            # Technical patterns
            'version': re.compile(r'(version|ver|v\.?)\s*:?\s*(\d+(?:\.\d+)*)', re.IGNORECASE),
            'system_name': re.compile(r'(System|Application|Software|Platform)\s+[A-Z][A-Za-z0-9\s]+', re.IGNORECASE),
            'requirement_id': re.compile(r'(REQ|REQUIREMENT|FR|NFR)[\-\s]*(\d+)', re.IGNORECASE),
        }
    
    def _compile_structure_patterns(self):
        """Compile document structure detection patterns."""
        self.structure_patterns = {
            'title_case': re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$'),
            'upper_case': re.compile(r'^[A-Z\s]+$'),
            'has_periods': re.compile(r'\.'),
            'starts_number': re.compile(r'^\d+'),
            'starts_letter': re.compile(r'^[A-Za-z]'),
        }
    
    # Helper methods for table detection
    def _is_table_separator(self, line: str) -> bool:
        """Check if line is a table separator."""
        return bool(self.patterns['table_separator'].match(line) or 
                   self.patterns['table_border'].match(line))
    
    def _is_table_row(self, line: str) -> bool:
        """Check if line looks like a table row."""
        # Look for multiple columns separated by pipes, tabs, or multiple spaces
        separators = len(re.findall(r'[\|]|\t|\s{3,}', line))
        return separators >= 2
    
    def _has_multiple_columns(self, line: str) -> bool:
        """Check if line has multiple columns."""
        columns = self._split_table_columns(line)
        return len(columns) >= 2
    
    def _could_be_table_content(self, line: str) -> bool:
        """Check if line could be table content."""
        # Avoid false positives for regular paragraphs
        if len(line) > 100:
            return False
        if '|' in line or '\t' in line:
            return True
        # Check for multiple spaced segments
        segments = re.split(r'\s{3,}', line)
        return len(segments) >= 2
    
    def _split_table_columns(self, line: str) -> List[str]:
        """Split line into table columns."""
        # Try pipe separator first
        if '|' in line:
            columns = [col.strip() for col in line.split('|') if col.strip()]
        # Try tab separator
        elif '\t' in line:
            columns = [col.strip() for col in line.split('\t') if col.strip()]
        # Try multiple spaces
        else:
            columns = [col.strip() for col in re.split(r'\s{3,}', line) if col.strip()]
        
        return columns
    
    def _classify_table_type(self, headers: List[str], rows: List[List[str]]) -> str:
        """Classify table type based on content."""
        header_text = ' '.join(headers).lower()
        
        if any(financial in header_text for financial in ['amount', 'cost', 'price', 'budget', 'revenue']):
            return 'financial'
        elif any(schedule in header_text for schedule in ['date', 'time', 'schedule', 'deadline']):
            return 'schedule'
        elif len(headers) <= 2:
            return 'list'
        else:
            return 'data'
    
    # Helper methods for section analysis
    def _classify_line(self, line: str) -> SectionType:
        """Classify line into section type."""
        if not line:
            return SectionType.PARAGRAPH
        
        # Check for headers
        if (line.isupper() and len(line) > 5 and len(line) < 100 and 
            not '.' in line):
            return SectionType.HEADER
        
        if (self.patterns['numbered_header'].match(line) or 
            self.patterns['lettered_header'].match(line)):
            return SectionType.SUBHEADER
        
        if line.istitle() and len(line) > 10 and len(line) < 80:
            return SectionType.SUBHEADER
        
        # Check for lists
        if (self.patterns['bullet_list'].match(line) or 
            self.patterns['numbered_list'].match(line)):
            return SectionType.LIST_ITEM
        
        # Check for references
        if self.patterns['reference_line'].match(line):
            return SectionType.REFERENCE
        
        return SectionType.PARAGRAPH
    
    def _determine_section_level(self, line: str, section_type: SectionType) -> int:
        """Determine hierarchical level of section."""
        if section_type == SectionType.TITLE:
            return 0
        elif section_type == SectionType.HEADER:
            return 1
        elif section_type == SectionType.SUBHEADER:
            # Check for numbered levels
            numbered_match = self.patterns['numbered_header'].match(line)
            if numbered_match:
                # Count dots to determine level
                dots = line[:20].count('.')
                return min(dots, 4)  # Cap at level 4
            return 2
        else:
            return 3
    
    def _establish_section_hierarchy(self, sections: List[DocumentSection]):
        """Establish parent-child relationships between sections."""
        for i, section in enumerate(sections):
            # Find parent (previous section with lower level)
            for j in range(i - 1, -1, -1):
                parent_candidate = sections[j]
                if parent_candidate.level < section.level:
                    section.parent_id = parent_candidate.section_id
                    parent_candidate.children_ids.append(section.section_id)
                    break
    
    # Entity extraction methods
    def _extract_universal_entities(self, text: str, start_id: int) -> List[DocumentEntity]:
        """Extract universal entities (email, phone, dates)."""
        entities = []
        entity_id = start_id
        
        # Extract emails
        for match in self.entity_patterns['email'].finditer(text):
            entities.append(DocumentEntity(
                entity_id=f"entity_{entity_id}",
                entity_type=EntityType.EMAIL,
                text=match.group(),
                context=self._get_context(text, match.start(), match.end()),
                confidence=0.95,
                start_position=match.start(),
                end_position=match.end()
            ))
            entity_id += 1
        
        # Extract phone numbers
        for match in self.entity_patterns['phone'].finditer(text):
            entities.append(DocumentEntity(
                entity_id=f"entity_{entity_id}",
                entity_type=EntityType.PHONE,
                text=match.group(),
                context=self._get_context(text, match.start(), match.end()),
                confidence=0.90,
                start_position=match.start(),
                end_position=match.end()
            ))
            entity_id += 1
        
        # Extract dates
        for match in self.entity_patterns['date'].finditer(text):
            entities.append(DocumentEntity(
                entity_id=f"entity_{entity_id}",
                entity_type=EntityType.DATE,
                text=match.group(),
                context=self._get_context(text, match.start(), match.end()),
                confidence=0.85,
                start_position=match.start(),
                end_position=match.end()
            ))
            entity_id += 1
        
        return entities
    
    def _extract_government_entities(self, text: str, start_id: int) -> List[DocumentEntity]:
        """Extract government-specific entities."""
        entities = []
        entity_id = start_id
        
        # Extract departments
        for match in self.entity_patterns['department'].finditer(text):
            entities.append(DocumentEntity(
                entity_id=f"entity_{entity_id}",
                entity_type=EntityType.DEPARTMENT,
                text=match.group(),
                context=self._get_context(text, match.start(), match.end()),
                confidence=0.90,
                start_position=match.start(),
                end_position=match.end()
            ))
            entity_id += 1
        
        # Extract reference numbers
        for match in self.entity_patterns['reference_number'].finditer(text):
            entities.append(DocumentEntity(
                entity_id=f"entity_{entity_id}",
                entity_type=EntityType.REFERENCE_NUMBER,
                text=match.group(2),  # Extract just the number part
                context=self._get_context(text, match.start(), match.end()),
                confidence=0.95,
                start_position=match.start(),
                end_position=match.end()
            ))
            entity_id += 1
        
        # Extract official titles
        for match in self.entity_patterns['official_title'].finditer(text):
            entities.append(DocumentEntity(
                entity_id=f"entity_{entity_id}",
                entity_type=EntityType.OFFICIAL_TITLE,
                text=match.group(),
                context=self._get_context(text, match.start(), match.end()),
                confidence=0.85,
                start_position=match.start(),
                end_position=match.end()
            ))
            entity_id += 1
        
        return entities
    
    def _extract_technical_entities(self, text: str, start_id: int) -> List[DocumentEntity]:
        """Extract technical document entities."""
        entities = []
        entity_id = start_id
        
        # Extract version numbers
        for match in self.entity_patterns['version'].finditer(text):
            entities.append(DocumentEntity(
                entity_id=f"entity_{entity_id}",
                entity_type=EntityType.VERSION_NUMBER,
                text=match.group(2),  # Extract just the version number
                context=self._get_context(text, match.start(), match.end()),
                confidence=0.95,
                start_position=match.start(),
                end_position=match.end()
            ))
            entity_id += 1
        
        # Extract system names
        for match in self.entity_patterns['system_name'].finditer(text):
            entities.append(DocumentEntity(
                entity_id=f"entity_{entity_id}",
                entity_type=EntityType.SYSTEM_NAME,
                text=match.group(),
                context=self._get_context(text, match.start(), match.end()),
                confidence=0.85,
                start_position=match.start(),
                end_position=match.end()
            ))
            entity_id += 1
        
        # Extract requirement IDs
        for match in self.entity_patterns['requirement_id'].finditer(text):
            entities.append(DocumentEntity(
                entity_id=f"entity_{entity_id}",
                entity_type=EntityType.REQUIREMENT_ID,
                text=match.group(),
                context=self._get_context(text, match.start(), match.end()),
                confidence=0.90,
                start_position=match.start(),
                end_position=match.end()
            ))
            entity_id += 1
        
        return entities
    
    def _extract_cr_entities(self, text: str, start_id: int) -> List[DocumentEntity]:
        """Extract change request specific entities."""
        # Combine technical and some government patterns for CR documents
        entities = []
        entities.extend(self._extract_technical_entities(text, start_id))
        
        # Add CR-specific patterns here if needed
        return entities
    
    def _get_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """Get surrounding context for an entity."""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end].strip()
    
    def _calculate_complexity_score(self, word_count: int, section_count: int, 
                                  table_count: int, entity_count: int) -> float:
        """Calculate document complexity score."""
        # Normalize components
        word_score = min(1.0, word_count / 10000)  # Normalize to 10k words
        section_score = min(1.0, section_count / 50)  # Normalize to 50 sections
        table_score = min(1.0, table_count / 20)  # Normalize to 20 tables
        entity_score = min(1.0, entity_count / 100)  # Normalize to 100 entities
        
        # Weighted average
        complexity = (word_score * 0.3 + section_score * 0.3 + 
                     table_score * 0.2 + entity_score * 0.2)
        
        return round(complexity, 3)
