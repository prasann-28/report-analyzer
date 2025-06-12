"""
Analysis API routes for company filing analysis.
Handles document processing, LLM analysis, and result storage.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Body
from pydantic import BaseModel, Field, validator

from config import get_settings
from core.sec_data_fetcher import SECDataFetcher, SECFiling, CompanyInfo
from core.document_processor import DocumentProcessor, ProcessedDocument
from core.llm_analyzer import LLMAnalyzer, CompanyAnalysis
from core.vector_store import VectorStore
from utils.exceptions import APIError


settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["Analysis"])


# Request/Response models
class AnalysisRequest(BaseModel):
    """Request model for company analysis."""
    company_identifier: str = Field(..., description="Company ticker or CIK")
    filing_types: Optional[List[str]] = Field(default=["10-K"], description="Filing types to analyze")
    max_filings: int = Field(default=1, ge=1, le=5, description="Maximum number of filings")
    include_vector_indexing: bool = Field(default=True, description="Whether to index for search")
    force_reprocess: bool = Field(default=False, description="Force reprocessing even if cached")


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""
    companies: List[str] = Field(..., min_items=1, max_items=10, description="Company tickers")
    filing_type: str = Field(default="10-K", description="Filing type to analyze")
    max_concurrent: int = Field(default=3, ge=1, le=5, description="Max concurrent analyses")


class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    request_id: str
    company_name: str
    ticker: str
    filing_id: str
    form_type: str
    filing_date: str
    analysis: CompanyAnalysis
    processing_time_seconds: float
    status: str = "completed"


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""
    batch_id: str
    total_companies: int
    completed: int
    failed: int
    results: List[AnalysisResponse]
    processing_time_seconds: float


# Dependency injection (these would be imported from main.py in practice)
from main import get_sec_fetcher, get_document_processor, get_llm_analyzer, get_vector_store


@router.post("/company", response_model=AnalysisResponse)
async def analyze_company(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    sec_fetcher: SECDataFetcher = Depends(get_sec_fetcher),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    llm_analyzer: LLMAnalyzer = Depends(get_llm_analyzer),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Analyze SEC filings for a specific company.
    
    This endpoint performs comprehensive analysis including:
    - Fetching latest SEC filings
    - Document processing and section extraction
    - LLM-powered risk and financial analysis
    - Vector indexing for semantic search
    """
    start_time = datetime.now()
    request_id = f"analysis_{request.company_identifier}_{int(start_time.timestamp())}"
    
    try:
        logger.info(f"Starting analysis for {request.company_identifier}", 
                   extra={"request_id": request_id})
        
        # Step 1: Get company information
        async with sec_fetcher:
            company_info = await sec_fetcher.get_company_info(request.company_identifier)
            logger.info(f"Found company: {company_info.name} ({company_info.ticker})")
            
            # Step 2: Fetch recent filings
            filings = await sec_fetcher.get_company_filings(
                identifier=request.company_identifier,
                form_types=request.filing_types,
                limit=request.max_filings
            )
            
            if not filings:
                raise HTTPException(
                    status_code=404,
                    detail=f"No {request.filing_types} filings found for {request.company_identifier}"
                )
            
            # Process the most recent filing
            target_filing = filings[0]
            logger.info(f"Processing filing {target_filing.accession_number}")
            
            # Step 3: Download filing content
            filing_content = await sec_fetcher.download_filing_content(target_filing)
        
        # Step 4: Process document
        filing_metadata = {
            "accession_number": target_filing.accession_number,
            "company_name": target_filing.company_name,
            "ticker": target_filing.ticker,
            "cik": target_filing.cik,
            "form_type": target_filing.form_type,
            "filing_date": target_filing.filing_date,
            "report_date": target_filing.report_date,
            "url": target_filing.url
        }
        
        processed_doc = await doc_processor.process_filing(filing_content, filing_metadata)
        logger.info(f"Processed document: {processed_doc.total_words} words, "
                   f"{len(processed_doc.sections)} sections")
        
        # Step 5: Perform LLM analysis
        analysis = await llm_analyzer.analyze_document(processed_doc)
        logger.info(f"Analysis completed: {len(analysis.risk_factors)} risks, "
                   f"{len(analysis.financial_metrics)} metrics")
        
        # Step 6: Index for search (if requested)
        if request.include_vector_indexing:
            background_tasks.add_task(
                _index_document_background,
                vector_store,
                processed_doc,
                request_id
            )
        
        # Prepare response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            request_id=request_id,
            company_name=company_info.name,
            ticker=company_info.ticker,
            filing_id=target_filing.accession_number,
            form_type=target_filing.form_type,
            filing_date=target_filing.filing_date,
            analysis=analysis,
            processing_time_seconds=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {request.company_identifier}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchAnalysisResponse)
async def analyze_companies_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    sec_fetcher: SECDataFetcher = Depends(get_sec_fetcher),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    llm_analyzer: LLMAnalyzer = Depends(get_llm_analyzer),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Analyze multiple companies in batch.
    
    Processes multiple companies concurrently with rate limiting.
    Suitable for portfolio analysis or sector comparison.
    """
    start_time = datetime.now()
    batch_id = f"batch_{int(start_time.timestamp())}"
    
    try:
        logger.info(f"Starting batch analysis for {len(request.companies)} companies",
                   extra={"batch_id": batch_id})
        
        # Process companies concurrently
        analysis_tasks = []
        semaphore = asyncio.Semaphore(request.max_concurrent)
        
        async def analyze_single_company(ticker: str) -> Optional[AnalysisResponse]:
            async with semaphore:
                try:
                    # Create individual request
                    individual_request = AnalysisRequest(
                        company_identifier=ticker,
                        filing_types=[request.filing_type],
                        max_filings=1,
                        include_vector_indexing=True
                    )
                    
                    # Perform analysis
                    result = await _perform_single_analysis(
                        individual_request,
                        sec_fetcher,
                        doc_processor,
                        llm_analyzer,
                        vector_store
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {ticker}: {str(e)}")
                    return None
        
        # Execute all tasks
        tasks = [analyze_single_company(ticker) for ticker in request.companies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed_count += 1
                logger.error(f"Task failed: {str(result)}")
            elif result is None:
                failed_count += 1
            else:
                successful_results.append(result)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchAnalysisResponse(
            batch_id=batch_id,
            total_companies=len(request.companies),
            completed=len(successful_results),
            failed=failed_count,
            results=successful_results,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.get("/companies/{ticker}/filings")
async def get_company_filings(
    ticker: str,
    form_types: Optional[List[str]] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    sec_fetcher: SECDataFetcher = Depends(get_sec_fetcher)
):
    """
    Get available SEC filings for a company.
    
    Returns metadata about available filings without processing them.
    Useful for exploring what filings are available before analysis.
    """
    try:
        async with sec_fetcher:
            filings = await sec_fetcher.get_company_filings(
                identifier=ticker,
                form_types=form_types,
                limit=limit
            )
            
            return {
                "ticker": ticker.upper(),
                "total_filings": len(filings),
                "filings": [
                    {
                        "accession_number": f.accession_number,
                        "form_type": f.form_type,
                        "filing_date": f.filing_date,
                        "report_date": f.report_date,
                        "size": f.size,
                        "description": f.document_description
                    }
                    for f in filings
                ]
            }
            
    except Exception as e:
        logger.error(f"Failed to get filings for {ticker}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve filings: {str(e)}"
        )


@router.post("/compare")
async def compare_companies(
    companies: List[str] = Body(..., min_items=2, max_items=5),
    filing_type: str = Body(default="10-K"),
    comparison_aspects: List[str] = Body(default=["risks", "metrics", "trends"]),
    sec_fetcher: SECDataFetcher = Depends(get_sec_fetcher),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    llm_analyzer: LLMAnalyzer = Depends(get_llm_analyzer)
):
    """
    Compare multiple companies across specified aspects.
    
    Performs side-by-side analysis of companies focusing on:
    - Risk factor comparison
    - Financial metrics comparison
    - Trend analysis comparison
    """
    try:
        # Perform analysis for all companies
        batch_request = BatchAnalysisRequest(
            companies=companies,
            filing_type=filing_type,
            max_concurrent=3
        )
        
        # This would call the batch analysis internally
        batch_result = await analyze_companies_batch(
            batch_request,
            BackgroundTasks(),
            sec_fetcher,
            doc_processor,
            llm_analyzer,
            None  # vector_store not needed for comparison
        )
        
        # Extract and compare specific aspects
        comparison_data = {}
        
        for aspect in comparison_aspects:
            if aspect == "risks":
                comparison_data["risk_comparison"] = _compare_risks(batch_result.results)
            elif aspect == "metrics":
                comparison_data["metrics_comparison"] = _compare_metrics(batch_result.results)
            elif aspect == "trends":
                comparison_data["trends_comparison"] = _compare_trends(batch_result.results)
        
        return {
            "companies": companies,
            "filing_type": filing_type,
            "comparison_date": datetime.now().isoformat(),
            "comparison_data": comparison_data,
            "summary": _generate_comparison_summary(batch_result.results)
        }
        
    except Exception as e:
        logger.error(f"Company comparison failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


# Helper functions
async def _perform_single_analysis(
    request: AnalysisRequest,
    sec_fetcher: SECDataFetcher,
    doc_processor: DocumentProcessor,
    llm_analyzer: LLMAnalyzer,
    vector_store: VectorStore
) -> AnalysisResponse:
    """Perform analysis for a single company (helper for batch processing)."""
    # This would contain the core logic from analyze_company
    # Extracted to avoid duplication
    pass


async def _index_document_background(
    vector_store: VectorStore,
    processed_doc: ProcessedDocument,
    request_id: str
):
    """Background task to index document for search."""
    try:
        success = await vector_store.index_document(processed_doc)
        if success:
            logger.info(f"Document indexed successfully for request {request_id}")
        else:
            logger.warning(f"Document indexing failed for request {request_id}")
    except Exception as e:
        logger.error(f"Background indexing failed: {str(e)}")


def _compare_risks(results: List[AnalysisResponse]) -> Dict[str, Any]:
    """Compare risk factors across companies."""
    risk_comparison = {}
    
    for result in results:
        risks_by_category = {}
        for risk in result.analysis.risk_factors:
            if risk.category not in risks_by_category:
                risks_by_category[risk.category] = []
            risks_by_category[risk.category].append({
                "description": risk.description,
                "impact_level": risk.impact_level,
                "likelihood": risk.likelihood
            })
        
        risk_comparison[result.ticker] = {
            "total_risks": len(result.analysis.risk_factors),
            "risk_score": result.analysis.overall_risk_score,
            "risks_by_category": risks_by_category
        }
    
    return risk_comparison


def _compare_metrics(results: List[AnalysisResponse]) -> Dict[str, Any]:
    """Compare financial metrics across companies."""
    metrics_comparison = {}
    
    for result in results:
        company_metrics = {}
        for metric in result.analysis.financial_metrics:
            company_metrics[metric.metric_name] = {
                "value": metric.value,
                "unit": metric.unit,
                "period": metric.period
            }
        
        metrics_comparison[result.ticker] = company_metrics
    
    return metrics_comparison


def _compare_trends(results: List[AnalysisResponse]) -> Dict[str, Any]:
    """Compare trends across companies."""
    trends_comparison = {}
    
    for result in results:
        company_trends = {}
        for trend in result.analysis.trends:
            company_trends[trend.metric] = {
                "direction": trend.direction,
                "magnitude": trend.magnitude,
                "timeframe": trend.timeframe
            }
        
        trends_comparison[result.ticker] = company_trends
    
    return trends_comparison


def _generate_comparison_summary(results: List[AnalysisResponse]) -> Dict[str, Any]:
    """Generate high-level comparison summary."""
    return {
        "total_companies_analyzed": len(results),
        "average_risk_score": sum(r.analysis.overall_risk_score for r in results) / len(results),
        "companies_by_risk_level": {
            "high_risk": [r.ticker for r in results if r.analysis.overall_risk_score > 7],
            "medium_risk": [r.ticker for r in results if 4 <= r.analysis.overall_risk_score <= 7],
            "low_risk": [r.ticker for r in results if r.analysis.overall_risk_score < 4]
        }
    }