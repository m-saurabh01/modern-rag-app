"""
Tests for Document Structure Analyzer.

This module tests the comprehensive document analysis functionality including:
- Document structure detection and hierarchy mapping
- Table identification and parsing
- Entity extraction for various document types
- Metadata generation and analysis
"""

import pytest
from typing import List, Dict
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.document_analyzer import (
    DocumentAnalyzer, DocumentStructure, DocumentSection, TableStructure,
    DocumentEntity, SectionType, EntityType
)


class TestDocumentAnalyzer:
    """Test suite for DocumentAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a DocumentAnalyzer instance for testing."""
        return DocumentAnalyzer()
    
    @pytest.fixture
    def sample_government_doc(self):
        """Sample government document text."""
        return """
DEPARTMENT OF HEALTH AND HUMAN SERVICES
OFFICE OF THE SECRETARY

MEMORANDUM

TO: All Regional Directors
FROM: Secretary of Health
DATE: March 15, 2024
REF: HHS-2024-001

SUBJECT: NEW HEALTH POLICY IMPLEMENTATION

I. INTRODUCTION

This memorandum establishes new guidelines for health policy implementation across all regions.

II. POLICY DETAILS

The following requirements must be met:
1. Complete staff training by April 30, 2024
2. Submit compliance reports monthly
3. Implement new procedures within 60 days

Contact Information:
Email: policy@hhs.gov
Phone: (555) 123-4567

Director of Policy Implementation
Office of the Secretary
"""
    
    @pytest.fixture
    def sample_technical_doc(self):
        """Sample technical manual text."""
        return """
System Installation Guide
Version 2.1.0

Table of Contents
1. System Requirements
2. Installation Process
3. Configuration

SYSTEM REQUIREMENTS

Hardware Requirements:
CPU     | RAM   | Storage
---------|-------|--------
Intel i5 | 8GB   | 500GB
AMD Ryzen| 16GB  | 1TB

Software Requirements:
- Operating System: Windows 10 or later
- Framework: .NET 6.0
- Database: SQL Server 2019

INSTALLATION PROCESS

1. Download the installer from the official website
2. Run setup.exe as administrator
3. Follow the installation wizard

For technical support, contact: support@system.com

REQ-001: System must start within 30 seconds
REQ-002: Database connections must be secure
"""
    
    @pytest.fixture
    def sample_table_doc(self):
        """Sample document with various table formats."""
        return """
Financial Report Q1 2024

Revenue Summary:

| Department | Q1 Revenue | Growth |
|------------|------------|---------|
| Sales      | $250,000   | +15%   |
| Marketing  | $180,000   | +8%    |
| Operations | $320,000   | +22%   |

Budget Allocation:

Category        Amount      Percentage
---------      --------    -----------
Personnel      $500,000         45%
Equipment      $200,000         18%
Training       $150,000         14%
Miscellaneous  $250,000         23%

Key Performance Indicators:
• Customer Satisfaction: 92%
• Employee Retention: 87%
• Market Share Growth: 5.2%
"""
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert hasattr(analyzer, 'patterns')
        assert hasattr(analyzer, 'entity_patterns')
        assert hasattr(analyzer, 'structure_patterns')
        assert analyzer.logger is not None
    
    def test_document_type_detection(self, analyzer):
        """Test document type detection."""
        # Government document
        gov_text = "Department of Health circular regarding new policy implementation"
        assert analyzer._detect_document_type(gov_text) == 'government'
        
        # Technical manual
        tech_text = "User manual for system configuration and troubleshooting guide"
        assert analyzer._detect_document_type(tech_text) == 'technical_manual'
        
        # Change request
        cr_text = "Change request CR-123 for feature enhancement and bug fix implementation"
        assert analyzer._detect_document_type(cr_text) == 'change_request'
        
        # Business report
        business_text = "Quarterly financial report analysis with budget summary"
        assert analyzer._detect_document_type(business_text) == 'business_report'
        
        # General document
        general_text = "This is a simple document without specific indicators"
        assert analyzer._detect_document_type(general_text) == 'general'
    
    def test_title_extraction(self, analyzer):
        """Test document title extraction."""
        # Title in first line
        text1 = "IMPORTANT POLICY DOCUMENT\n\nThis document contains important information..."
        title1 = analyzer._extract_document_title(text1)
        assert title1 == "IMPORTANT POLICY DOCUMENT"
        
        # Title case
        text2 = "System Installation Guide\nVersion 1.0\n\nThis guide will help you..."
        title2 = analyzer._extract_document_title(text2)
        assert title2 == "System Installation Guide"
        
        # No clear title
        text3 = "This is just a paragraph without a clear title structure."
        title3 = analyzer._extract_document_title(text3)
        assert "This is just a paragraph" in title3
    
    def test_table_detection_pipe_format(self, analyzer):
        """Test table detection with pipe-separated format."""
        table_text = """
| Name    | Age | Department |
|---------|-----|------------|
| John    | 30  | Sales      |
| Sarah   | 25  | Marketing  |
"""
        tables = analyzer._detect_tables(table_text)
        assert len(tables) == 1
        
        table = tables[0]
        assert len(table.headers) == 3
        assert table.headers == ['Name', 'Age', 'Department']
        assert len(table.rows) == 2
        assert table.rows[0] == ['John', '30', 'Sales']
        assert table.column_count == 3
        assert table.row_count == 2
    
    def test_table_detection_space_format(self, analyzer):
        """Test table detection with space-separated format."""
        table_text = """
Department        Revenue     Growth
---------        --------    --------
Sales            $250,000    +15%
Marketing        $180,000    +8%
Operations       $320,000    +22%
"""
        tables = analyzer._detect_tables(table_text)
        assert len(tables) == 1
        
        table = tables[0]
        assert len(table.headers) >= 2  # Should detect multiple columns
        assert len(table.rows) >= 2     # Should detect data rows
    
    def test_section_analysis(self, analyzer, sample_government_doc):
        """Test document section analysis."""
        tables = []  # No tables in this doc
        sections = analyzer._analyze_sections(sample_government_doc, tables)
        
        assert len(sections) > 0
        
        # Check for different section types
        section_types = [s.section_type for s in sections]
        assert SectionType.HEADER in section_types or SectionType.SUBHEADER in section_types
        
        # Check hierarchy levels
        levels = [s.level for s in sections]
        assert min(levels) >= 0
        assert max(levels) <= 4
    
    def test_government_entity_extraction(self, analyzer, sample_government_doc):
        """Test government entity extraction."""
        entities = analyzer._extract_government_entities(sample_government_doc, 0)
        
        # Should find department
        dept_entities = [e for e in entities if e.entity_type == EntityType.DEPARTMENT]
        assert len(dept_entities) > 0
        
        # Should find reference number
        ref_entities = [e for e in entities if e.entity_type == EntityType.REFERENCE_NUMBER]
        assert len(ref_entities) > 0
        
        # Should find official title
        title_entities = [e for e in entities if e.entity_type == EntityType.OFFICIAL_TITLE]
        assert len(title_entities) > 0
    
    def test_technical_entity_extraction(self, analyzer, sample_technical_doc):
        """Test technical entity extraction."""
        entities = analyzer._extract_technical_entities(sample_technical_doc, 0)
        
        # Should find version number
        version_entities = [e for e in entities if e.entity_type == EntityType.VERSION_NUMBER]
        assert len(version_entities) > 0
        
        # Should find requirement IDs
        req_entities = [e for e in entities if e.entity_type == EntityType.REQUIREMENT_ID]
        assert len(req_entities) > 0
    
    def test_universal_entity_extraction(self, analyzer):
        """Test universal entity extraction."""
        text = """
        Contact us at info@company.com or call (555) 123-4567.
        The meeting is scheduled for 03/15/2024 at 2:00 PM.
        Alternative date: 2024-03-20.
        """
        
        entities = analyzer._extract_universal_entities(text, 0)
        
        # Should find email
        email_entities = [e for e in entities if e.entity_type == EntityType.EMAIL]
        assert len(email_entities) > 0
        assert "info@company.com" in [e.text for e in email_entities]
        
        # Should find phone
        phone_entities = [e for e in entities if e.entity_type == EntityType.PHONE]
        assert len(phone_entities) > 0
        
        # Should find dates
        date_entities = [e for e in entities if e.entity_type == EntityType.DATE]
        assert len(date_entities) >= 1
    
    def test_complete_document_analysis(self, analyzer, sample_government_doc):
        """Test complete document analysis workflow."""
        result = analyzer.analyze_document(sample_government_doc, "test_doc_1")
        
        # Check result structure
        assert isinstance(result, DocumentStructure)
        assert result.document_id == "test_doc_1"
        assert result.document_type in ['government', 'business_report', 'general']
        assert result.title != "Untitled Document"
        
        # Check components
        assert isinstance(result.sections, list)
        assert isinstance(result.tables, list)
        assert isinstance(result.entities, list)
        assert isinstance(result.metadata, dict)
        
        # Check metadata
        assert 'word_count' in result.metadata
        assert 'section_count' in result.metadata
        assert 'complexity_score' in result.metadata
        assert result.metadata['word_count'] > 0
    
    def test_table_classification(self, analyzer):
        """Test table type classification."""
        # Financial table
        financial_headers = ['Department', 'Revenue', 'Cost', 'Profit']
        financial_rows = [['Sales', '$100k', '$50k', '$50k']]
        table_type = analyzer._classify_table_type(financial_headers, financial_rows)
        assert table_type == 'financial'
        
        # Schedule table
        schedule_headers = ['Task', 'Start Date', 'End Date']
        schedule_rows = [['Setup', '2024-01-01', '2024-01-15']]
        table_type = analyzer._classify_table_type(schedule_headers, schedule_rows)
        assert table_type == 'schedule'
        
        # Simple list
        list_headers = ['Name', 'Value']
        list_rows = [['Item1', 'Value1']]
        table_type = analyzer._classify_table_type(list_headers, list_rows)
        assert table_type == 'list'
        
        # Data table
        data_headers = ['ID', 'Name', 'Status', 'Priority']
        data_rows = [['1', 'Task1', 'Active', 'High']]
        table_type = analyzer._classify_table_type(data_headers, data_rows)
        assert table_type == 'data'
    
    def test_section_hierarchy(self, analyzer):
        """Test section hierarchy establishment."""
        text = """
MAIN TITLE

1. FIRST SECTION
This is the first section content.

1.1 First Subsection
Content of first subsection.

1.2 Second Subsection
Content of second subsection.

2. SECOND SECTION
This is the second section content.

2.1 Another Subsection
More content here.
"""
        tables = []
        sections = analyzer._analyze_sections(text, tables)
        
        # Should establish proper hierarchy
        analyzer._establish_section_hierarchy(sections)
        
        # Check parent-child relationships
        parents = [s for s in sections if s.parent_id is not None]
        assert len(parents) > 0
        
        # Check children lists
        with_children = [s for s in sections if len(s.children_ids) > 0]
        assert len(with_children) > 0
    
    def test_complexity_calculation(self, analyzer):
        """Test document complexity score calculation."""
        # Simple document
        simple_score = analyzer._calculate_complexity_score(100, 3, 0, 5)
        assert 0.0 <= simple_score <= 1.0
        
        # Complex document
        complex_score = analyzer._calculate_complexity_score(5000, 25, 10, 50)
        assert complex_score > simple_score
        assert 0.0 <= complex_score <= 1.0
        
        # Maximum complexity
        max_score = analyzer._calculate_complexity_score(10000, 50, 20, 100)
        assert max_score > complex_score
        assert max_score <= 1.0
    
    def test_context_extraction(self, analyzer):
        """Test entity context extraction."""
        text = "Please contact the Department of Health for more information about the policy."
        start_pos = text.find("Department of Health")
        end_pos = start_pos + len("Department of Health")
        
        context = analyzer._get_context(text, start_pos, end_pos, 20)
        assert "contact" in context
        assert "Department of Health" in context
        assert "policy" in context
        assert len(context) <= len(text)
    
    def test_empty_document_handling(self, analyzer):
        """Test handling of empty or minimal documents."""
        # Empty document
        empty_result = analyzer.analyze_document("", "empty_doc")
        assert empty_result.document_type == 'general'
        assert len(empty_result.sections) == 0
        assert len(empty_result.tables) == 0
        assert len(empty_result.entities) == 0
        
        # Minimal document
        minimal_text = "Short document."
        minimal_result = analyzer.analyze_document(minimal_text, "minimal_doc")
        assert minimal_result.document_type == 'general'
        assert minimal_result.metadata['word_count'] == 2
    
    def test_multiple_tables_detection(self, analyzer, sample_table_doc):
        """Test detection of multiple tables in one document."""
        tables = analyzer._detect_tables(sample_table_doc)
        
        # Should detect multiple tables
        assert len(tables) >= 1
        
        # Check table properties
        for table in tables:
            assert len(table.headers) > 0
            assert len(table.rows) > 0
            assert table.column_count > 0
            assert table.row_count > 0
    
    def test_line_classification(self, analyzer):
        """Test line classification for section types."""
        # Header (all caps)
        header_line = "IMPORTANT SECTION HEADER"
        assert analyzer._classify_line(header_line) == SectionType.HEADER
        
        # Numbered header
        numbered_line = "1. Introduction"
        assert analyzer._classify_line(numbered_line) == SectionType.SUBHEADER
        
        # Bullet list
        bullet_line = "• First item in list"
        assert analyzer._classify_line(bullet_line) == SectionType.LIST_ITEM
        
        # Numbered list
        numbered_list_line = "1. First numbered item"
        assert analyzer._classify_line(numbered_list_line) == SectionType.LIST_ITEM
        
        # Regular paragraph
        paragraph_line = "This is a regular paragraph with normal text."
        assert analyzer._classify_line(paragraph_line) == SectionType.PARAGRAPH
    
    def test_performance_with_large_document(self, analyzer):
        """Test analyzer performance with larger documents."""
        # Create a larger synthetic document
        large_doc_parts = []
        
        # Add title and headers
        large_doc_parts.append("COMPREHENSIVE POLICY DOCUMENT")
        large_doc_parts.append("")
        
        # Add multiple sections
        for i in range(10):
            large_doc_parts.append(f"{i+1}. SECTION {i+1}")
            large_doc_parts.append("")
            
            # Add content paragraphs
            for j in range(5):
                large_doc_parts.append(f"This is paragraph {j+1} of section {i+1}. " * 10)
            large_doc_parts.append("")
        
        # Add a table
        large_doc_parts.extend([
            "Data Summary:",
            "| Item | Value | Status |",
            "|------|-------|--------|",
            "| A    | 100   | Active |",
            "| B    | 200   | Pending|",
            ""
        ])
        
        # Add contact info
        large_doc_parts.append("Contact: admin@department.gov or (555) 123-4567")
        
        large_text = '\n'.join(large_doc_parts)
        
        # Analyze the document
        result = analyzer.analyze_document(large_text, "large_doc")
        
        # Check that analysis completed successfully
        assert result is not None
        assert result.processing_time > 0
        assert len(result.sections) > 5
        assert len(result.tables) >= 1
        assert len(result.entities) > 0
        assert result.metadata['complexity_score'] > 0


if __name__ == "__main__":
    # Run basic tests
    analyzer = DocumentAnalyzer()
    
    # Test with sample government document
    sample_text = """
DEPARTMENT OF EDUCATION
POLICY MEMORANDUM

TO: All School Districts
FROM: State Superintendent
DATE: March 1, 2024
REF: ED-2024-005

SUBJECT: New Academic Standards Implementation

1. OVERVIEW
This memorandum outlines the implementation of new academic standards.

2. REQUIREMENTS
Schools must:
• Complete teacher training by June 2024
• Submit implementation plans by April 15, 2024
• Conduct student assessments quarterly

Contact: standards@education.gov
Phone: (555) 987-6543

Superintendent of Public Instruction
State Department of Education
"""
    
    print("Testing Document Structure Analyzer...")
    result = analyzer.analyze_document(sample_text, "test_sample")
    
    print(f"Document Type: {result.document_type}")
    print(f"Title: {result.title}")
    print(f"Sections: {len(result.sections)}")
    print(f"Tables: {len(result.tables)}")
    print(f"Entities: {len(result.entities)}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print(f"Complexity Score: {result.metadata['complexity_score']}")
    
    print("\nEntity Details:")
    for entity in result.entities:
        print(f"  {entity.entity_type.value}: {entity.text}")
    
    print("\nSection Details:")
    for section in result.sections:
        print(f"  Level {section.level}: {section.section_type.value} - {section.title}")
    
    print("\nAnalysis completed successfully!")
