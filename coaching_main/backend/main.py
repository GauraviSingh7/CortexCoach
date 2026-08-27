"""
FastAPI Backend for AI Coaching Observer
FULLY CORRECTED VERSION - Ready to use
"""
import os
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.core.orchestrator import CoachingObserverSystem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Coaching Observer API",
    description="Real-time coaching session analysis and feedback",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[CoachingObserverSystem] = None


# Request/Response Models
class SessionStartRequest(BaseModel):
    session_type: str = "live"          # "live" | "file" | "replay"
    device_index: Optional[int] = None
    coach_speaker_id: Optional[str] = None  # File/replay mode: "A" or "B"
    transcript_path: Optional[str] = None   # Replay mode only


class SessionStartResponse(BaseModel):
    session_id: str
    status: str


@app.on_event("startup")
async def startup_event():
    """Initialize the coaching observer system on startup"""
    global orchestrator
    
    assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not assemblyai_key:
        logger.warning("⚠️ ASSEMBLYAI_API_KEY not found in environment variables")
    
    if not gemini_key:
        logger.warning("⚠️ GEMINI_API_KEY not found - reports will use local analysis only")
    
    orchestrator = CoachingObserverSystem(
        assemblyai_key=assemblyai_key,
        gemini_key=gemini_key
    )
    
    logger.info("✅ AI Coaching Observer API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global orchestrator
    if orchestrator and orchestrator.session_active:
        try:
            await orchestrator.stop_session()
        except:
            pass
    logger.info("👋 AI Coaching Observer API shutting down")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Coaching Observer API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "session_active": orchestrator.session_active if orchestrator else False
    }


@app.get("/devices/audio")
async def get_audio_devices():
    """Get available audio input devices"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=500, detail="System not initialized")
        
        devices = orchestrator.get_available_audio_devices()
        return {"devices": devices}
    
    except HTTPException:
        # Raised deliberately above - it carries the right status and
        # message, so it must not be re-wrapped as a 500 below.
        raise
    except Exception as e:
        logger.error(f"Error getting audio devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """Start a new coaching session"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=500, detail="System not initialized")
        
        if orchestrator.session_active:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A session is already running "
                    f"({orchestrator.state.session_type if orchestrator.state else 'live'}). "
                    "Stop it before starting another."
                ),
            )
        
        # Replay mode reads a stored transcript, so it needs no API key.
        if request.session_type != "replay" and not os.getenv("ASSEMBLYAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="ASSEMBLYAI_API_KEY not configured. Please set it in your .env file."
            )
        
        session_id = await orchestrator.start_session(
            session_type=request.session_type,
            device_index=request.device_index,
            file_path=request.transcript_path,
            coach_speaker_id=request.coach_speaker_id
        )
        
        logger.info(f"✅ Session started: {session_id}")
        
        return SessionStartResponse(
            session_id=session_id,
            status="started"
        )
        
    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"Error starting session: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    except HTTPException:
        # Raised deliberately above - it carries the right status and
        # message, so it must not be re-wrapped as a 500 below.
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/session/start/file")
async def start_file_session(
    file: UploadFile = File(...),
    coach_speaker_id: Optional[str] = None
):
    """Start a session by uploading an audio file"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=500, detail="System not initialized")
        
        if orchestrator.session_active:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A session is already running "
                    f"({orchestrator.state.session_type if orchestrator.state else 'live'}). "
                    "Stop it before starting another."
                ),
            )
        
        # Save uploaded file temporarily
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Start session with file
        session_id = await orchestrator.start_session(
            session_type="file",
            file_path=str(file_path),
            coach_speaker_id=coach_speaker_id
        )
        
        return {
            "session_id": session_id,
            "status": "started",
            "type": "file",
            "filename": file.filename
        }
    
    except HTTPException:
        # Raised deliberately above - it carries the right status and
        # message, so it must not be re-wrapped as a 500 below.
        raise
    except Exception as e:
        logger.error(f"Error starting file session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/stop")
async def stop_session():
    """Stop the current coaching session and generate report.

    Idempotent: if there is no active session but we already produced a
    report for the most recent one, return it again instead of 400.
    Avoids the retry-cascade UX when the frontend times out and re-posts.
    """
    try:
        if not orchestrator:
            raise HTTPException(status_code=500, detail="System not initialized")

        if not orchestrator.session_active:
            if orchestrator.last_report is not None:
                logger.info("Stop called with no active session — returning cached last report")
                return {
                    "status": "stopped",
                    "report": orchestrator.last_report.model_dump(),
                    "report_file": None,
                    "cached": True
                }
            raise HTTPException(status_code=400, detail="No active session")

        report = await orchestrator.stop_session()
        
        # Save report to file
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_file = reports_dir / f"{report.session_id}.json"
        with open(report_file, "w") as f:
            f.write(report.model_dump_json(indent=2))
        
        # Also save as latest
        latest_file = reports_dir / "coaching_analysis_full_report.json"
        with open(latest_file, "w") as f:
            f.write(report.model_dump_json(indent=2))
        
        return {
            "status": "stopped",
            "report": report.model_dump(),
            "report_file": str(report_file)
        }
    
    except HTTPException:
        # Raised deliberately above - it carries the right status and
        # message, so it must not be re-wrapped as a 500 below.
        raise
    except Exception as e:
        logger.error(f"Error stopping session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/status")
async def get_session_status():
    """Get current session status"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    state = orchestrator.state if orchestrator.session_active else None

    return {
        "active": orchestrator.session_active,
        "session_id": orchestrator.session_id if orchestrator.session_active else None,
        "chunks_processed": len(orchestrator.session_data.get("chunks", [])) if orchestrator.session_active else 0,
        "session_type": state.session_type if state else None,
        # Replay/file sources run out; the dashboard uses this to stop
        # polling and prompt for the report. Always False for live mode.
        "source_finished": bool(state.source_finished) if state else False,
        # A live capture thread that died leaves the session looking healthy
        # while nothing is recorded. Surface it instead.
        "capture_warning": (
            getattr(orchestrator.audio_processor, "capture_warning", None)
            if orchestrator.session_active and orchestrator.audio_processor
            else None
        ),
        "capture_error": (
            getattr(orchestrator.audio_processor, "stream_error", None)
            if orchestrator.session_active and orchestrator.audio_processor
            else None
        ),
    }


@app.websocket("/ws/feedback")
async def websocket_feedback(websocket: WebSocket):
    """WebSocket endpoint for real-time coaching feedback"""
    await websocket.accept()
    
    if not orchestrator:
        await websocket.close(code=1011, reason="System not initialized")
        return
    
    # Add client to orchestrator's websocket clients
    orchestrator.websocket_clients.add(websocket)
    logger.info(f"✅ WebSocket client connected. Total clients: {len(orchestrator.websocket_clients)}")
    
    try:
        # Keep connection alive and receive messages if needed
        while True:
            try:
                # Wait for messages from client (e.g., ping/pong)
                data = await websocket.receive_text()
                logger.debug(f"Received from client: {data}")
                
                # Echo back or handle client messages
                if data == "ping":
                    await websocket.send_text("pong")
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    
    finally:
        # Remove client on disconnect
        orchestrator.websocket_clients.discard(websocket)
        logger.info(f"❌ WebSocket client disconnected. Remaining clients: {len(orchestrator.websocket_clients)}")


@app.get("/model-status")
async def get_model_status():
    """Get the status of all ML models"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=500, detail="System not initialized")
        
        # Returns per-model state ("trained" / "heuristic" / "unavailable")
        # with the blocking reason attached. The previous version reported
        # "all_loaded: true" whenever the wrapper classes constructed, which
        # they always did - even with no weights in memory.
        return orchestrator.inference_engine.get_model_status()
    
    except HTTPException:
        # Raised deliberately above - it carries the right status and
        # message, so it must not be re-wrapped as a 500 below.
        raise
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
