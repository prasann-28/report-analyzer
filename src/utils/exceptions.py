"""
Custom exceptions for the AI Financial Analyzer application.
Provides specific error types for different components and failure modes.
"""

from typing import Optional, Dict, Any


class AIFinancialAnalyzerError(Exception):
    """Base exception for all application errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(AIFinancialAnalyzerError):
    """Raised when there are configuration issues."""
    pass


class APIError(AIFinancialAnalyzerError):
    """General API error."""
    pass


class SECAPIError(AIFinancialAnalyzerError):
    """Raised when SEC API operations fail."""
    pass


class RateLimitError(SECAPIError):
    """Raised when rate limits are exceeded."""
    pass


class DocumentProcessingError(AIFinancialAnalyzerError):
    """Raised when document processing fails."""
    pass


class LLMAnalysisError(AIFinancialAnalyzerError):
    """Raised when LLM analysis operations fail."""
    pass


class VectorStoreError(AIFinancialAnalyzerError):
    """Raised when vector store operations fail."""
    pass


class PineconeConnectionError(VectorStoreError):
    """Raised when Pinecone connection fails."""
    pass


class EmbeddingError(VectorStoreError):
    """Raised when embedding generation fails."""
    pass


class ValidationError(AIFinancialAnalyzerError):
    """Raised when data validation fails."""
    pass


class FileStorageError(AIFinancialAnalyzerError):
    """Raised when file storage operations fail."""
    pass


class CacheError(AIFinancialAnalyzerError):
    """Raised when cache operations fail."""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    pass


class ResourceNotFoundError(APIError):
    """Raised when requested resources are not found."""
    pass


class ServiceUnavailableError(APIError):
    """Raised when external services are unavailable."""
    pass


class DataQualityError(AIFinancialAnalyzerError):
    """Raised when data quality issues are detected."""
    pass


class ProcessingTimeoutError(AIFinancialAnalyzerError):
    """Raised when processing operations timeout."""
    pass