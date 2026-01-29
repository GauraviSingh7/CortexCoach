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
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.core.orchestrator import CoachingObserverSystem
from backend.schemas.data_models import SessionReport

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
    session_type: str = "live"
    device_index: Optional[int] = None


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
            raise HTTPException(status_code=400, detail="Session already active")
        
        # Check for API key
        if not os.getenv("ASSEMBLYAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="ASSEMBLYAI_API_KEY not configured. Please set it in your .env file."
            )
        
        session_id = await orchestrator.start_session(
            session_type=request.session_type,
            device_index=request.device_index
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
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/session/start/file")
async def start_file_session(file: UploadFile = File(...)):
    """Start a session by uploading an audio file"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=500, detail="System not initialized")
        
        if orchestrator.session_active:
            raise HTTPException(status_code=400, detail="Session already active")
        
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
            file_path=str(file_path)
        )
        
        return {
            "session_id": session_id,
            "status": "started",
            "type": "file",
            "filename": file.filename
        }
    
    except Exception as e:
        logger.error(f"Error starting file session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/stop")
async def stop_session():
    """Stop the current coaching session and generate report"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=500, detail="System not initialized")
        
        if not orchestrator.session_active:
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
    
    except Exception as e:
        logger.error(f"Error stopping session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/status")
async def get_session_status():
    """Get current session status"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    return {
        "active": orchestrator.session_active,
        "session_id": orchestrator.session_id if orchestrator.session_active else None,
        "chunks_processed": len(orchestrator.session_data.get("chunks", [])) if orchestrator.session_active else 0
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
        
        model_status = orchestrator.inference_engine.get_model_status()
        
        return {
            "models": model_status,
            "all_loaded": all(status == "loaded" for status in model_status.values())
        }
    
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
