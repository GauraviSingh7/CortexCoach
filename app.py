import streamlit as st
import asyncio
import threading
import time
from datetime import datetime
from pathlib import Path
import sys
import logging
import os

os.makedirs("logs", exist_ok=True)
# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

sys.path.append(str(project_root / "core"))


from config.settings import *
from core.coaching_rag_system import (
    CoachingRAGSystem,
    MultimodalContext,
    GROWPhase,
    VARKType
)
from core.session_manager import SessionManager
from ui.streamlit_app import StreamlitCoachingApp
from server import RealTimeCoachingServer
from models.inference_pipeline import MultimodalInferencePipeline

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["file"], mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultimodalCoachingApp:
    """Main application class that coordinates all components"""
    
    def __init__(self):
        self.coaching_system = None
        self.session_manager = None
        self.websocket_server = None
        self.streamlit_app = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Create necessary directories
            self._create_directories()

            # Initialize coaching RAG system
            logger.info("Initializing coaching RAG system...")
            self.coaching_system = CoachingRAGSystem(
                gemini_api_key=GEMINI_API_KEY,
                chroma_persist_dir=str(CHROMA_DB_PATH)
            )

            # ✅ Initialize inference pipeline BEFORE session manager
            logger.info("Initializing inference pipeline...")
            self.model_pipeline = MultimodalInferencePipeline(MODEL_PATHS)

            # Initialize session manager
            logger.info("Initializing session manager...")
            self.session_manager = SessionManager(
                coaching_system=self.coaching_system,
                inference_pipeline=self.model_pipeline,  # ✅ Now it exists
                timeout_minutes=SESSION_CONFIG["timeout_minutes"]
            )

            # Initialize WebSocket server
            logger.info("Initializing WebSocket server...")
            self.websocket_server = RealTimeCoachingServer(
                coaching_system=self.coaching_system,
                session_manager=self.session_manager,
                host=WEBSOCKET_HOST,
                port=WEBSOCKET_PORT
            )

            # Initialize Streamlit app
            logger.info("Initializing Streamlit app...")
            self.streamlit_app = StreamlitCoachingApp(
                coaching_system=self.coaching_system,
                session_manager=self.session_manager
            )

            logger.info("✅ All components initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise

    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            CHROMA_DB_PATH,
            LOGGING_CONFIG["file"].parent,
            MODEL_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run_streamlit(self):
        """Run the Streamlit interface"""
        try:
            logger.info(f"Starting Streamlit app on {STREAMLIT_HOST}:{STREAMLIT_PORT}")
            self.streamlit_app.run()
        except Exception as e:
            logger.error(f"Failed to start Streamlit app: {e}")
            raise
    
    def run_websocket_server(self):
        """Run the WebSocket server for real-time coaching"""
        try:
            logger.info(f"Starting WebSocket server on {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
            asyncio.run(self.websocket_server.start_server())
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    def run_both(self):
        """Run both Streamlit and WebSocket server concurrently"""
        def run_websocket():
            """Run WebSocket server in a separate thread"""
            asyncio.run(self.websocket_server.start_server())
        
        # Start WebSocket server in background thread
        websocket_thread = threading.Thread(target=run_websocket, daemon=True)
        websocket_thread.start()
        
        # Give server time to start
        time.sleep(2)
        
        # Run Streamlit in main thread
        self.run_streamlit()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal AI Coaching System")
    parser.add_argument(
        "--mode", 
        choices=["streamlit", "websocket", "both"],
        default="streamlit",
        help="Run mode: streamlit only, websocket only, or both"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    try:
        # Initialize the main application
        app = MultimodalCoachingApp()
        
        # Run based on selected mode
        if args.mode == "streamlit":
            app.run_streamlit()
        elif args.mode == "websocket":
            app.run_websocket_server()
        elif args.mode == "both":
            app.run_both()
            
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main()