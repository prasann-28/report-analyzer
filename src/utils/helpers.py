"""
Utility functions and helpers for the AI Financial Analyzer.
Provides common functionality used across different components.
"""

import asyncio
import functools
import re
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


def retry_async(max_retries: int = 3, backoff_factor: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Async retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff multiplier
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    sleep_time = backoff_factor * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {sleep_time}s: {str(e)}")
                    await asyncio.sleep(sleep_time)
            
            raise last_exception
            
        return wrapper
    return decorator


def rate_limit(calls: int, period: float):
    """
    Rate limiting decorator.
    
    Args:
        calls: Number of calls allowed
        period: Time period in seconds
    """
    def decorator(func):
        calls_times = []
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls outside the period
            calls_times[:] = [call_time for call_time in calls_times if now - call_time < period]
            
            # Check if we can make another call
            if len(calls_times) >= calls:
                sleep_time = period - (now - calls_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    calls_times.pop(0)
            
            # Record this call
            calls_times.append(now)
            
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
    
    # Normalize line breaks
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # Remove excessive line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip and return
    return text.strip()


def extract_tables(soup: BeautifulSoup) -> List[pd.DataFrame]:
    """
    Extract tables from HTML content.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        List of DataFrames
    """
    tables = []
    
    for table in soup.find_all('table'):
        try:
            # Convert to DataFrame
            df = pd.read_html(str(table))[0]
            
            # Basic cleaning
            df = df.dropna(how='all')  # Remove empty rows
            df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
            
            if not df.empty and len(df) > 1:
                tables.append(df)
                
        except Exception as e:
            logger.debug(f"Failed to parse table: {str(e)}")
            continue
    
    return tables


def truncate_text(text: str, max_tokens: int = 4000, encoding: str = "cl100k_base") -> str:
    """
    Truncate text to fit within token limits.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        encoding: Tokenizer encoding to use
        
    Returns:
        Truncated text
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding)
        tokens = enc.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate to max_tokens and decode
        truncated_tokens = tokens[:max_tokens]
        return enc.decode(truncated_tokens)
        
    except ImportError:
        # Fallback to character-based truncation
        # Rough estimate: 1 token â‰ˆ 4 characters
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        
        return text[:max_chars] + "..."


def extract_financial_numbers(text: str) -> List[Dict[str, Any]]:
    """
    Extract financial numbers and amounts from text.
    
    Args:
        text: Text to extract numbers from
        
    Returns:
        List of financial numbers with context
    """
    financial_patterns = [
        # Dollar amounts
        r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(million|billion|trillion|thousand)?',
        # Percentages
        r'(\d+(?:\.\d+)?)\s*%',
        # Revenue/income patterns
        r'(?:revenue|income|earnings|profit|loss).*?\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(million|billion|trillion|thousand)?',
    ]
    
    financial_numbers = []
    
    for pattern in financial_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract context around the match
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            
            financial_numbers.append({
                "value": match.group(1),
                "unit": match.group(2) if len(match.groups()) > 1 else None,
                "context": context.strip(),
                "position": match.start()
            })
    
    return financial_numbers


def normalize_company_name(name: str) -> str:
    """
    Normalize company name for consistent matching.
    
    Args:
        name: Company name to normalize
        
    Returns:
        Normalized company name
    """
    if not name:
        return ""
    
    # Convert to lowercase
    normalized = name.lower()
    
    # Remove common suffixes
    suffixes = [
        r'\s+inc\.?$', r'\s+corp\.?$', r'\s+corporation$', r'\s+company$',
        r'\s+co\.?$', r'\s+ltd\.?$', r'\s+limited$', r'\s+llc$',
        r'\s+&\s+co\.?$', r'\s+enterprises?$'
    ]
    
    for suffix in suffixes:
        normalized = re.sub(suffix, '', normalized)
    
    # Remove punctuation and extra spaces
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized.strip()


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to word sets
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def format_currency(amount: Union[int, float], currency: str = "USD") -> str:
    """
    Format monetary amounts with appropriate scale and currency.
    
    Args:
        amount: Monetary amount
        currency: Currency code
        
    Returns:
        Formatted currency string
    """
    if amount >= 1_000_000_000:
        return f"{currency} {amount / 1_000_000_000:.2f}B"
    elif amount >= 1_000_000:
        return f"{currency} {amount / 1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"{currency} {amount / 1_000:.2f}K"
    else:
        return f"{currency} {amount:.2f}"


def validate_ticker(ticker: str) -> bool:
    """
    Validate stock ticker format.
    
    Args:
        ticker: Stock ticker to validate
        
    Returns:
        True if valid ticker format
    """
    if not ticker:
        return False
    
    # Basic ticker validation: 1-5 letters, optionally with dots
    pattern = r'^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$'
    return bool(re.match(pattern, ticker.upper()))


def extract_section_content(html_content: str, section_pattern: str) -> Optional[str]:
    """
    Extract specific section content from HTML.
    
    Args:
        html_content: Full HTML content
        section_pattern: Regex pattern for section identification
        
    Returns:
        Extracted section content or None
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text()
    
    # Find section boundaries
    pattern = re.compile(section_pattern, re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text_content)
    
    if not match:
        return None
    
    start_pos = match.start()
    
    # Find next section or end of document
    next_section_patterns = [
        r'ITEM\s+\d+[A-Za-z]?[\.\-â€“â€”]',
        r'PART\s+[IVX]+',
        r'SIGNATURE'
    ]
    
    end_pos = len(text_content)
    for next_pattern in next_section_patterns:
        next_match = re.search(next_pattern, text_content[start_pos + 100:], re.IGNORECASE)
        if next_match:
            end_pos = start_pos + 100 + next_match.start()
            break
    
    section_content = text_content[start_pos:end_pos]
    return clean_text(section_content)


def parse_filing_date(date_str: str) -> Optional[datetime]:
    """
    Parse filing date from various string formats.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        Parsed datetime or None
    """
    if not date_str:
        return None
    
    date_formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y%m%d"
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    return None


def calculate_processing_stats(start_time: datetime, items_processed: int) -> Dict[str, Any]:
    """
    Calculate processing statistics.
    
    Args:
        start_time: Processing start time
        items_processed: Number of items processed
        
    Returns:
        Dictionary with processing statistics
    """
    end_time = datetime.now()
    duration = end_time - start_time
    
    return {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "items_processed": items_processed,
        "items_per_second": items_processed / duration.total_seconds() if duration.total_seconds() > 0 else 0,
        "average_time_per_item": duration.total_seconds() / items_processed if items_processed > 0 else 0
    }


def create_error_context(operation: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create standardized error context for logging and debugging.
    
    Args:
        operation: Name of the operation that failed
        details: Additional context details
        
    Returns:
        Error context dictionary
    """
    return {
        "operation": operation,
        "timestamp": datetime.now().isoformat(),
        "details": details,
        "error_id": f"{operation}_{int(time.time())}"
    }