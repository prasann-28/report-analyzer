"""
FastAPI main application for AI Financial Report Analyzer.
Provides REST API endpoints for financial document analysis and search.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

from config import get_settings
from core.sec_data_fetcher import SECDataFetcher, SECFiling
from core.document_processor import DocumentProcessor, ProcessedDocument
from core.llm_analyzer import LLMAnalyzer, CompanyAnalysis
from core.vector_store import VectorStore, SearchResult
from api.routes.analysis import router as analysis_router
from api.routes.search import router as search_router
from api.routes.health import router as health_router
from utils.exceptions import APIError, SECAPIError, DocumentProcessingError, LLMAnalysisError
import structlog


# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
settings = get_settings()


# Application state management
class AppState:
    """Application state container."""
    def __init__(self):
        self.sec_fetcher: Optional[SECDataFetcher] = None
        self.document_processor: Optional[DocumentProcessor] = None
        self.llm_analyzer: Optional[LLMAnalyzer] = None
        self.vector_store: Optional[VectorStore] = None
        self.startup_time: Optional[datetime] = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AI Financial Report Analyzer")
    
    try:
        # Initialize Sentry if configured
        if settings.sentry_dsn:
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                integrations=[FastApiIntegration()],
                traces_sample_rate=0.1,
            )
            logger.info("Sentry monitoring initialized")
        
        # Initialize components
        app_state.sec_fetcher = SECDataFetcher()
        app_state.document_processor = DocumentProcessor()
        app_state.llm_analyzer = LLMAnalyzer()
        app_state.vector_store = VectorStore()
        app_state.startup_time = datetime.now()
        
        logger.info("All components initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    # Shutdown
    logger.info("Shutting down AI Financial Report Analyzer")
    
    # Cleanup resources
    if app_state.sec_fetcher and hasattr(app_state.sec_fetcher, 'session'):
        if app_state.sec_fetcher.session:
            await app_state.sec_fetcher.session.close()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI-powered SEC filing analysis and semantic search platform",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Exception handlers
@app.exception_handler(SECAPIError)
async def sec_api_exception_handler(request, exc):
    logger.error(f"SEC API error: {str(exc)}")
    return JSONResponse(
        status_code=503,
        content={"error": "SEC API unavailable", "detail": str(exc)}
    )


@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request, exc):
    logger.error(f"Document processing error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"error": "Document processing failed", "detail": str(exc)}
    )


@app.exception_handler(LLMAnalysisError)
async def llm_analysis_exception_handler(request, exc):
    logger.error(f"LLM analysis error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Analysis failed", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )


# Dependency injection
async def get_sec_fetcher() -> SECDataFetcher:
    """Get SEC data fetcher instance."""
    if not app_state.sec_fetcher:
        raise HTTPException(status_code=503, detail="SEC fetcher not initialized")
    return app_state.sec_fetcher


async def get_document_processor() -> DocumentProcessor:
    """Get document processor instance."""
    if not app_state.document_processor:
        raise HTTPException(status_code=503, detail="Document processor not initialized")
    return app_state.document_processor


async def get_llm_analyzer() -> LLMAnalyzer:
    """Get LLM analyzer instance."""
    if not app_state.llm_analyzer:
        raise HTTPException(status_code=503, detail="LLM analyzer not initialized")
    return app_state.llm_analyzer


async def get_vector_store() -> VectorStore:
    """Get vector store instance."""
    if not app_state.vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return app_state.vector_store


# Include routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "docs_url": "/api/v1/docs",
        "uptime_seconds": (
            datetime.now() - app_state.startup_time
        ).total_seconds() if app_state.startup_time else 0
    }


# Status endpoint
@app.get("/api/v1/status")
async def get_status():
    """Get application status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.app_version,
        "components": {
            "sec_fetcher": app_state.sec_fetcher is not None,
            "document_processor": app_state.document_processor is not None,
            "llm_analyzer": app_state.llm_analyzer is not None,
            "vector_store": app_state.vector_store is not None
        }
    }


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        workers=1 if settings.debug else settings.workers
    )