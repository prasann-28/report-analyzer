"""
Health check routes for monitoring system status.
Provides endpoints for health monitoring, readiness checks, and system metrics.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from config import get_settings
from core.vector_store import VectorStore, IndexStats
from utils.exceptions import VectorStoreError


settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str


class DetailedHealthStatus(BaseModel):
    """Detailed health status with component checks."""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str
    components: Dict[str, Dict[str, Any]]
    system_metrics: Dict[str, Any]


class ReadinessStatus(BaseModel):
    """Readiness status for Kubernetes probes."""
    ready: bool
    timestamp: str
    components: Dict[str, bool]


# Store startup time
startup_time = datetime.now()


@router.get("/", response_model=HealthStatus)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns basic application health status.
    Used for simple uptime monitoring.
    """
    current_time = datetime.now()
    uptime = (current_time - startup_time).total_seconds()
    
    return HealthStatus(
        status="healthy",
        timestamp=current_time.isoformat(),
        uptime_seconds=uptime,
        version=settings.app_version
    )


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check():
    """
    Detailed health check with component status.
    
    Performs comprehensive health checks on all system components
    including external dependencies and resource usage.
    """
    current_time = datetime.now()
    uptime = (current_time - startup_time).total_seconds()
    
    # Check all components
    components = {}
    
    # Check OpenAI API
    components["openai"] = await _check_openai_health()
    
    # Check Pinecone
    components["pinecone"] = await _check_pinecone_health()
    
    # Check SEC API
    components["sec_api"] = await _check_sec_api_health()
    
    # System metrics
    system_metrics = await _get_system_metrics()
    
    # Determine overall status
    overall_status = "healthy"
    if any(comp["status"] == "unhealthy" for comp in components.values()):
        overall_status = "unhealthy"
    elif any(comp["status"] == "degraded" for comp in components.values()):
        overall_status = "degraded"
    
    return DetailedHealthStatus(
        status=overall_status,
        timestamp=current_time.isoformat(),
        uptime_seconds=uptime,
        version=settings.app_version,
        components=components,
        system_metrics=system_metrics
    )


@router.get("/ready", response_model=ReadinessStatus)
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    
    Checks if the application is ready to serve traffic.
    Returns 200 if ready, 503 if not ready.
    """
    current_time = datetime.now()
    
    # Check critical components for readiness
    components = {
        "openai": await _is_openai_ready(),
        "pinecone": await _is_pinecone_ready(),
        "configuration": _is_configuration_ready()
    }
    
    all_ready = all(components.values())
    
    if not all_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return ReadinessStatus(
        ready=all_ready,
        timestamp=current_time.isoformat(),
        components=components
    )


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    
    Simple check to verify the application is alive.
    Should only fail if the application is completely broken.
    """
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@router.get("/metrics")
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint.
    
    Provides application metrics in a format suitable for monitoring.
    """
    try:
        # Get vector store metrics if available
        from main import app_state
        
        vector_metrics = {}
        if app_state.vector_store:
            try:
                index_stats = app_state.vector_store.get_index_stats()
                vector_metrics = {
                    "total_vectors": index_stats.total_vectors,
                    "index_fullness": index_stats.index_fullness,
                    "namespace_count": index_stats.namespace_count
                }
            except Exception:
                vector_metrics = {"error": "Unable to retrieve vector metrics"}
        
        uptime = (datetime.now() - startup_time).total_seconds()
        
        metrics = {
            "app_info": {
                "name": settings.app_name,
                "version": settings.app_version,
                "uptime_seconds": uptime
            },
            "vector_store": vector_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


# Component health check functions
async def _check_openai_health() -> Dict[str, Any]:
    """Check OpenAI API health."""
    try:
        import openai
        openai.api_key = settings.openai_api_key
        
        # Simple API call to check connectivity
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: openai.Model.list()
        )
        
        return {
            "status": "healthy",
            "response_time_ms": 0,  # Would measure actual response time
            "models_available": len(response.get("data", []))
        }
        
    except Exception as e:
        logger.warning(f"OpenAI health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": None
        }


async def _check_pinecone_health() -> Dict[str, Any]:
    """Check Pinecone health."""
    try:
        import pinecone
        
        pinecone.init(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment
        )
        
        # Check if we can list indexes
        indexes = pinecone.list_indexes()
        
        # Check if our specific index exists and is ready
        index_status = "not_found"
        if settings.pinecone_index_name in indexes:
            index_info = pinecone.describe_index(settings.pinecone_index_name)
            index_status = index_info.status.get("ready", False)
        
        return {
            "status": "healthy" if index_status else "degraded",
            "index_exists": settings.pinecone_index_name in indexes,
            "index_ready": index_status,
            "total_indexes": len(indexes)
        }
        
    except Exception as e:
        logger.warning(f"Pinecone health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _check_sec_api_health() -> Dict[str, Any]:
    """Check SEC API health."""
    try:
        import aiohttp
        
        # Simple request to SEC API
        async with aiohttp.ClientSession() as session:
            headers = {"User-Agent": settings.sec_user_agent}
            url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"  # Apple
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "response_code": response.status,
                        "response_time_ms": 0  # Would measure actual response time
                    }
                else:
                    return {
                        "status": "degraded",
                        "response_code": response.status,
                        "error": f"HTTP {response.status}"
                    }
                    
    except Exception as e:
        logger.warning(f"SEC API health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _get_system_metrics() -> Dict[str, Any]:
    """Get system resource metrics."""
    try:
        import psutil
        
        # CPU and memory metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "memory": {
                "usage_percent": memory.percent,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3)
            },
            "disk": {
                "usage_percent": disk.percent,
                "free_gb": disk.free / (1024**3),
                "total_gb": disk.total / (1024**3)
            }
        }
        
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


async def _is_openai_ready() -> bool:
    """Check if OpenAI is ready."""
    try:
        health = await _check_openai_health()
        return health["status"] == "healthy"
    except Exception:
        return False


async def _is_pinecone_ready() -> bool:
    """Check if Pinecone is ready."""
    try:
        health = await _check_pinecone_health()
        return health["status"] in ["healthy", "degraded"]
    except Exception:
        return False


def _is_configuration_ready() -> bool:
    """Check if configuration is valid."""
    try:
        # Check required configuration
        required_settings = [
            settings.openai_api_key,
            settings.pinecone_api_key,
            settings.pinecone_environment,
            settings.aws_access_key_id,
            settings.aws_secret_access_key
        ]
        
        return all(setting for setting in required_settings)
        
    except Exception:
        return False