"""
FastAPI REST API for RedactionTool
Provides programmatic access to redaction features
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import tempfile

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.pii.ensemble_detector import EnsembleDetector
from app.services.redaction.enhanced_redactor import EnhancedRedactor
from app.services.redaction.policy_manager import PolicyManager
from app.services.audit.audit_logger import AuditLogger

# Configuration from environment variables
API_KEY = os.getenv("API_KEY", "default-api-key-change-me")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))

# Initialize FastAPI app
app = FastAPI(
    title="RedactionTool API",
    description="Enterprise PII Redaction REST API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
detector = EnsembleDetector()
redactor = EnhancedRedactor()
audit_logger = AuditLogger(log_dir="audit_logs/api")

# Include extensions router
from api.extensions_router import router as extensions_router
app.include_router(extensions_router)


# ============================================================================
# Models
# ============================================================================

class EntityResponse(BaseModel):
    """Entity detection response"""
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    source: str


class RedactionRequest(BaseModel):
    """Redaction request body"""
    text: str
    policy: Optional[str] = None
    mode: str = "block"


class RedactionResponse(BaseModel):
    """Redaction response"""
    redacted_text: str
    entities_detected: int
    entities_by_type: dict
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str


# ============================================================================
# Security
# ============================================================================

def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key from header"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "RedactionTool API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )


@app.post("/detect", response_model=List[EntityResponse])
async def detect_entities(
    request: RedactionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect PII entities in text

    Args:
        request: Text to analyze
        api_key: API key for authentication

    Returns:
        List of detected entities
    """
    try:
        entities = detector.detect(request.text)

        return [
            EntityResponse(
                entity_type=e["entity_type"],
                text=e["text"],
                start=e["start"],
                end=e["end"],
                confidence=e.get("confidence", 0.0),
                source=e.get("source", "unknown")
            )
            for e in entities
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/redact", response_model=RedactionResponse)
async def redact_text(
    request: RedactionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Redact PII from text

    Args:
        request: Text to redact with optional policy
        api_key: API key for authentication

    Returns:
        Redacted text and statistics
    """
    try:
        start_time = datetime.now()

        # Detect entities
        entities = detector.detect(request.text)

        # Load policy if specified
        policy_dict = None
        if request.policy:
            policy_manager = PolicyManager()
            policy_manager.load_policy(f"policies/{request.policy}.yaml")
            policy_dict = policy_manager.name

        # Redact
        redacted_text = redactor.redact_text(
            request.text,
            entities,
            policy=policy_dict
        )

        # Calculate processing time
        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Entity breakdown
        entity_breakdown = {}
        for e in entities:
            entity_type = e["entity_type"]
            entity_breakdown[entity_type] = entity_breakdown.get(entity_type, 0) + 1

        # Log audit
        audit_logger.log_redaction_event(
            document_path="api_request",
            policy_name=request.policy,
            entities_detected=entities,
            actions_taken={e: request.mode for e in entity_breakdown.keys()},
            success=True,
            processing_time_ms=processing_time_ms
        )

        return RedactionResponse(
            redacted_text=redacted_text,
            entities_detected=len(entities),
            entities_by_type=entity_breakdown,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/redact/file")
async def redact_file(
    file: UploadFile = File(...),
    policy: Optional[str] = None,
    mode: str = "block",
    api_key: str = Depends(verify_api_key)
):
    """
    Redact PII from uploaded file

    Args:
        file: File to redact
        policy: Optional policy name
        mode: Redaction mode (block, mask, label)
        api_key: API key for authentication

    Returns:
        Redacted file
    """
    try:
        # Check file size
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)

        if file_size_mb > MAX_UPLOAD_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {MAX_UPLOAD_SIZE_MB} MB"
            )

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Process file (simplified - would use actual file loaders)
        text = contents.decode('utf-8', errors='ignore')

        # Detect and redact
        entities = detector.detect(text)
        redacted_text = redactor.redact_text(text, entities, policy=policy)

        # Save redacted file
        output_path = tmp_path.replace(Path(tmp_path).suffix, f"_redacted{Path(tmp_path).suffix}")
        with open(output_path, 'w') as f:
            f.write(redacted_text)

        # Return file
        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"redacted_{file.filename}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/policies")
async def list_policies(api_key: str = Depends(verify_api_key)):
    """List available redaction policies"""
    try:
        policies_dir = Path("policies")
        if not policies_dir.exists():
            return {"policies": []}

        policies = [
            p.stem for p in policies_dir.glob("*.yaml")
        ]

        return {"policies": policies}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats(api_key: str = Depends(verify_api_key)):
    """Get API statistics"""
    try:
        summary = audit_logger.get_session_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Error handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("Starting RedactionTool API...")
    print(f"API Key authentication enabled")
    print(f"CORS origins: {CORS_ORIGINS}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down RedactionTool API...")
    audit_logger.save_session_logs()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
