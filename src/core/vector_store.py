"""
Document Processor for parsing and chunking SEC filings.
Handles HTML parsing, text extraction, and document segmentation.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup, NavigableString
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from config import get_settings, SEC_SECTION_MAPPINGS
from utils.exceptions import DocumentProcessingError
from utils.helpers import clean_text, extract_tables


settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a section of a SEC filing."""
    section_id: str
    title: str
    content: str
    page_number: Optional[int] = None
    word_count: int = 0
    
    def __post_init__(self):
        self.word_count = len(self.content.split()) if self.content else 0


@dataclass
class ProcessedDocument:
    """Represents a fully processed SEC filing."""
    filing_id: str
    company_name: str
    ticker: str
    form_type: str
    filing_date: str
    sections: List[DocumentSection]
    chunks: List[Document]
    metadata: Dict
    raw_content: str
    
    @property
    def total_words(self) -> int:
        return sum(section.word_count for section in self.sections)


class DocumentProcessor:
    """
    Processes SEC filings by parsing HTML, extracting sections,
    and creating optimized chunks for LLM processing.
    """
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Common section patterns for different filing types
        self.section_patterns = {
            "10-K": self._get_10k_patterns(),
            "10-Q": self._get_10q_patterns(),
            "8-K": self._get_8k_patterns()
        }
    
    def _get_10k_patterns(self) -> Dict[str, re.Pattern]:
        """Get regex patterns for 10-K sections."""
        patterns = {}
        for section_id, title in SEC_SECTION_MAPPINGS.items():
            # Match patterns like "ITEM 1." or "Item 1A" or "PART I"
            pattern = rf"(?:ITEM|Item)\s*{re.escape(section_id)}\.?\s*[\-â€“â€”]?\s*{re.escape(title)}"
            patterns[section_id] = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        
        return patterns
    
    def _get_10q_patterns(self) -> Dict[str, re.Pattern]:
        """Get regex patterns for 10-Q sections."""
        # 10-Q has different section structure
        patterns = {
            "part1_item1": re.compile(r"PART\s*I.*?ITEM\s*1\.?\s*Financial\s*Statements", re.IGNORECASE),
            "part1_item2": re.compile(r"ITEM\s*2\.?\s*Management.*?Discussion", re.IGNORECASE),
            "part2_item1": re.compile(r"PART\s*II.*?ITEM\s*1\.?\s*Legal\s*Proceedings", re.IGNORECASE),
            "part2_item1a": re.compile(r"ITEM\s*1A\.?\s*Risk\s*Factors", re.IGNORECASE)
        }
        return patterns
    
    def _get_8k_patterns(self) -> Dict[str, re.Pattern]:
        """Get regex patterns for 8-K sections."""
        patterns = {
            "item1_01": re.compile(r"ITEM\s*1\.01.*?Entry.*?Agreement", re.IGNORECASE),
            "item2_02": re.compile(r"ITEM\s*2\.02.*?Results.*?Operations", re.IGNORECASE),
            "item8_01": re.compile(r"ITEM\s*8\.01.*?Other\s*Events", re.IGNORECASE)
        }
        return patterns
    
    async def process_filing(
        self,
        content: str,
        filing_metadata: Dict
    ) -> ProcessedDocument:
        """
        Process a complete SEC filing.
        
        Args:
            content: Raw filing content (HTML)
            filing_metadata: Filing metadata dict
            
        Returns:
            ProcessedDocument object
        """
        try:
            # Extract basic metadata
            filing_id = filing_metadata.get("accession_number", "unknown")
            company_name = filing_metadata.get("company_name", "")
            ticker = filing_metadata.get("ticker", "")
            form_type = filing_metadata.get("form_type", "")
            filing_date = filing_metadata.get("filing_date", "")
            
            logger.info(f"Processing filing {filing_id} for {company_name}")
            
            # Parse HTML and extract sections
            sections = await self._extract_sections(content, form_type)
            
            # Create document chunks for vector search
            chunks = await self._create_chunks(sections, filing_metadata)
            
            # Compile additional metadata
            metadata = {
                **filing_metadata,
                "total_sections": len(sections),
                "total_words": sum(s.word_count for s in sections),
                "processing_timestamp": pd.Timestamp.now().isoformat()
            }
            
            return ProcessedDocument(
                filing_id=filing_id,
                company_name=company_name,
                ticker=ticker,
                form_type=form_type,
                filing_date=filing_date,
                sections=sections,
                chunks=chunks,
                metadata=metadata,
                raw_content=content
            )
            
        except Exception as e:
            logger.error(f"Failed to process filing {filing_metadata.get('accession_number')}: {str(e)}")
            raise DocumentProcessingError(f"Document processing failed: {str(e)}")
    
    async def _extract_sections(self, content: str, form_type: str) -> List[DocumentSection]:
        """
        Extract sections from SEC filing content.
        
        Args:
            content: Raw HTML content
            form_type: Type of SEC form (10-K, 10-Q, etc.)
            
        Returns:
            List of DocumentSection objects
        """
        sections = []
        
        # Parse HTML
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove scripts, styles, and other non-content elements
        for element in soup(["script", "style", "meta", "link"]):
            element.decompose()
        
        # Extract text content
        text_content = soup.get_text()
        clean_content = clean_text(text_content)
        
        # Get patterns for this form type
        patterns = self.section_patterns.get(form_type, {})
        
        if not patterns:
            # If no specific patterns, create a single section
            logger.warning(f"No section patterns for form type {form_type}, treating as single section")
            sections.append(DocumentSection(
                section_id="full_document",
                title="Full Document",
                content=clean_content
            ))
            return sections
        
        # Find section boundaries
        section_boundaries = []
        for section_id, pattern in patterns.items():
            matches = list(pattern.finditer(clean_content))
            for match in matches:
                section_boundaries.append((match.start(), section_id, match.group()))
        
        # Sort by position in document
        section_boundaries.sort(key=lambda x: x[0])
        
        # Extract sections based on boundaries
        for i, (start_pos, section_id, title) in enumerate(section_boundaries):
            # Determine end position
            if i + 1 < len(section_boundaries):
                end_pos = section_boundaries[i + 1][0]
            else:
                end_pos = len(clean_content)
            
            # Extract section content
            section_content = clean_content[start_pos:end_pos].strip()
            
            # Remove the title from content to avoid duplication
            section_content = section_content[len(title):].strip()
            
            if section_content and len(section_content) > 100:  # Minimum content length
                sections.append(DocumentSection(
                    section_id=section_id,
                    title=SEC_SECTION_MAPPINGS.get(section_id, title),
                    content=section_content
                ))
        
        # If no sections found, use fallback extraction
        if not sections:
            sections = await self._fallback_section_extraction(clean_content, form_type)
        
        logger.info(f"Extracted {len(sections)} sections from {form_type} filing")
        return sections
    
    async def _fallback_section_extraction(self, content: str, form_type: str) -> List[DocumentSection]:
        """
        Fallback section extraction when pattern matching fails.
        
        Args:
            content: Clean text content
            form_type: Form type
            
        Returns:
            List of sections based on heuristics
        """
        sections = []
        
        # Split by common section indicators
        section_indicators = [
            r"ITEM\s+\d+[A-Za-z]?[\.\-â€“â€”]",
            r"PART\s+[IVX]+",
            r"TABLE\s+OF\s+CONTENTS",
            r"RISK\s+FACTORS",
            r"MANAGEMENT.*?DISCUSSION",
            r"FINANCIAL\s+STATEMENTS"
        ]
        
        combined_pattern = "|".join(f"({pattern})" for pattern in section_indicators)
        matches = list(re.finditer(combined_pattern, content, re.IGNORECASE | re.MULTILINE))
        
        if matches:
            for i, match in enumerate(matches):
                start_pos = match.start()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
                
                section_content = content[start_pos:end_pos].strip()
                section_title = match.group().strip()
                
                if len(section_content) > 200:  # Minimum meaningful content
                    sections.append(DocumentSection(
                        section_id=f"section_{i}",
                        title=section_title,
                        content=section_content
                    ))
        else:
            # Last resort: split into equal chunks
            chunk_size = max(5000, len(content) // 10)  # Target ~10 sections
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                if len(chunk) > 500:
                    sections.append(DocumentSection(
                        section_id=f"chunk_{i // chunk_size}",
                        title=f"Document Section {i // chunk_size + 1}",
                        content=chunk
                    ))
        
        return sections
    
    async def _create_chunks(
        self,
        sections: List[DocumentSection],
        filing_metadata: Dict
    ) -> List[Document]:
        """
        Create optimized chunks for vector search and LLM processing.
        
        Args:
            sections: List of document sections
            filing_metadata: Filing metadata
            
        Returns:
            List of LangChain Document objects
        """
        chunks = []
        
        for section in sections:
            # Split section content into chunks
            section_chunks = self.text_splitter.split_text(section.content)
            
            for i, chunk_text in enumerate(section_chunks):
                # Create metadata for each chunk
                chunk_metadata = {
                    **filing_metadata,
                    "section_id": section.section_id,
                    "section_title": section.title,
                    "chunk_index": i,
                    "total_chunks_in_section": len(section_chunks),
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text)
                }
                
                # Create LangChain Document
                doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                
                chunks.append(doc)
        
        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def extract_financial_tables(self, content: str) -> List[pd.DataFrame]:
        """
        Extract financial tables from HTML content.
        
        Args:
            content: HTML content
            
        Returns:
            List of pandas DataFrames
        """
        soup = BeautifulSoup(content, 'html.parser')
        tables = soup.find_all('table')
        
        financial_tables = []
        
        for table in tables:
            try:
                # Convert HTML table to DataFrame
                df = pd.read_html(str(table))[0]
                
                # Basic heuristics to identify financial tables
                if (len(df) > 2 and 
                    len(df.columns) > 1 and
                    any(col for col in df.columns if 
                        any(term in str(col).lower() for term in 
                            ['revenue', 'income', 'expense', 'assets', 'liabilities', '$', 'million']))):
                    
                    financial_tables.append(df)
                    
            except Exception as e:
                logger.debug(f"Could not parse table: {str(e)}")
                continue
        
        return financial_tables
    
    def extract_risk_factors(self, sections: List[DocumentSection]) -> List[str]:
        """
        Extract risk factors from document sections.
        
        Args:
            sections: List of document sections
            
        Returns:
            List of risk factor text
        """
        risk_factors = []
        
        for section in sections:
            if "risk" in section.title.lower():
                # Split risk factors by common patterns
                risk_patterns = [
                    r"(?:^|\n)\s*[\â€¢\-\*]\s*(.+?)(?=\n\s*[\â€¢\-\*]|\n\s*[A-Z][A-Z\s]+|$)",
                    r"(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\n\s*[A-Z][A-Z\s]+|$)",
                    r"(?:^|\n)\s*([A-Z][^.]*\.(?:[^.]*\.)*)",
                ]
                
                for pattern in risk_patterns:
                    matches = re.findall(pattern, section.content, re.MULTILINE | re.DOTALL)
                    for match in matches:
                        risk_text = clean_text(match.strip())
                        if len(risk_text) > 50:  # Minimum meaningful length
                            risk_factors.append(risk_text)
        
        return risk_factors
    
    def validate_processed_document(self, doc: ProcessedDocument) -> bool:
        """
        Validate processed document quality.
        
        Args:
            doc: ProcessedDocument to validate
            
        Returns:
            True if document passes validation
        """
        # Check minimum requirements
        if not doc.sections:
            logger.warning(f"No sections found in document {doc.filing_id}")
            return False
        
        if not doc.chunks:
            logger.warning(f"No chunks created for document {doc.filing_id}")
            return False
        
        # Check content quality
        total_words = doc.total_words
        if total_words < 1000:
            logger.warning(f"Document {doc.filing_id} has insufficient content ({total_words} words)")
            return False
        
        # Check for reasonable section distribution
        avg_section_length = total_words / len(doc.sections)
        if avg_section_length < 100:
            logger.warning(f"Document {doc.filing_id} has very short sections (avg: {avg_section_length} words)")
            return False
        
        return True