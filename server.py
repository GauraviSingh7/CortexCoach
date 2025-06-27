import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Dict, Set, Optional, Any
import uuid
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from core.coaching_rag_system import (
    CoachingRAGSystem, 
    MultimodalContext, 
    GROWPhase, 
    VARKType
)
from core.session_manager import SessionManager

logger = logging.getLogger(__name__)

class RealTimeCoachingServer:
    """WebSocket server for real-time multimodal coaching sessions"""
    
    def __init__(self, coaching_system: CoachingRAGSystem, session_manager: SessionManager, 
                 host: str = "localhost", port: int = 8765):
        self.coaching_system = coaching_system
        self.session_manager = session_manager
        self.host = host
        self.port = port
        
        # Track active connections
        self.active_connections: Set[websockets.WebSocketServerProtocol] = set()
        self.connection_sessions: Dict[websockets.WebSocketServerProtocol, str] = {}
        
        # Message handlers
        self.message_handlers = {
            "init_session": self._handle_init_session,
            "multimodal_input": self._handle_multimodal_input,
            "feedback": self._handle_feedback,
            "end_session": self._handle_end_session,
            "get_session_state": self._handle_get_session_state,
            "phase_transition": self._handle_phase_transition
        }
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10
        ):
            logger.info("WebSocket server started successfully")
            await asyncio.Future()  # Run forever
    
    async def handle_client(self, websocket, path):
        """Handle individual client connections"""
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        logger.info(f"New client connected: {client_ip}")
        
        # Add to active connections
        self.active_connections.add(websocket)
        
        try:
            async for message in websocket:
                await self._process_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_ip}")
        except Exception as e:
            logger.error(f"Error handling client {client_ip}: {e}")
            await self._send_error(websocket, f"Server error: {str(e)}")
        finally:
            # Clean up connection
            await self._cleanup_connection(websocket)
    
    async def _process_message(self, websocket, message: str):
        """Process incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type not in self.message_handlers:
                await self._send_error(websocket, f"Unknown message type: {message_type}")
                return
            
            # Call appropriate handler
            await self.message_handlers[message_type](websocket, data)
            
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self._send_error(websocket, f"Error processing message: {str(e)}")
    
    async def _handle_init_session(self, websocket, data: Dict):
        """Initialize a new coaching session"""
        try:
            user_id = data.get("user_id")
            session_config = data.get("config", {})
            
            if not user_id:
                await self._send_error(websocket, "user_id is required")
                return
            
            # Create new session
            session_id = await self.session_manager.create_session(
                user_id=user_id,
                initial_phase=GROWPhase(session_config.get("initial_phase", "goal")),
                config=session_config
            )
            
            # Associate connection with session
            self.connection_sessions[websocket] = session_id
            
            # Send session initialization response
            response = {
                "type": "session_initialized",
                "session_id": session_id,
                "user_id": user_id,
                "current_phase": "goal",
                "message": "Coaching session initialized successfully"
            }
            
            await websocket.send(json.dumps(response))
            logger.info(f"Session initialized: {session_id} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error initializing session: {e}")
            await self._send_error(websocket, f"Failed to initialize session: {str(e)}")
    
    async def _handle_multimodal_input(self, websocket, data: Dict):
        """Handle multimodal input and generate coaching response"""
        try:
            session_id = self.connection_sessions.get(websocket)
            if not session_id:
                await self._send_error(websocket, "No active session")
                return
            
            # Get session state
            session_state = await self.session_manager.get_session_state(session_id)
            if not session_state:
                await self._send_error(websocket, "Session not found")
                return
            
            # Extract multimodal data
            multimodal_data = data.get("multimodal_data", {})
            
            # Create MultimodalContext
            context = MultimodalContext(
                user_id=session_state["user_id"],
                session_id=session_id,
                timestamp=datetime.now(),
                grow_phase=GROWPhase(session_state["current_phase"]),
                utterance=multimodal_data.get("text", ""),
                
                # Emotion data
                facial_emotion=multimodal_data.get("facial_emotion", {}),
                voice_emotion=multimodal_data.get("voice_emotion", {}),
                text_sentiment=multimodal_data.get("text_sentiment", {}),
                
                # Learning style
                vark_type=VARKType(multimodal_data.get("vark_type", "visual")),
                vark_confidence=multimodal_data.get("vark_confidence", 0.5),
                
                # Behavioral indicators
                sarcasm_detected=multimodal_data.get("sarcasm_detected", False),
                sarcasm_confidence=multimodal_data.get("sarcasm_confidence", 0.0),
                interest_level=multimodal_data.get("interest_level", 0.5),
                digression_detected=multimodal_data.get("digression_detected", False),
                
                # Session context
                conversation_turn=session_state["conversation_turn"],
                previous_phase_completion=session_state.get("phase_completion", False),
                goal_clarity_score=session_state.get("goal_clarity_score", 0.0)
            )
            
            # Generate coaching response
            coaching_response = self.coaching_system.generate_coaching_response(context)
            
            # Update session state
            await self.session_manager.update_session_state(session_id, {
                "conversation_turn": session_state["conversation_turn"] + 1,
                "last_interaction": datetime.now().isoformat(),
                "last_utterance": context.utterance,
                "last_emotion": self.coaching_system._get_dominant_emotion(
                    context.facial_emotion, context.voice_emotion
                )
            })
            
            # Check for phase transition
            should_transition = await self._check_phase_transition(context, session_state)
            new_phase = None
            
            if should_transition:
                new_phase = await self._transition_phase(session_id, session_state)
            
            # Send response
            response = {
                "type": "coaching_response",
                "session_id": session_id,
                "response": coaching_response,
                "context": {
                    "current_phase": session_state["current_phase"],
                    "conversation_turn": session_state["conversation_turn"] + 1,
                    "dominant_emotion": self.coaching_system._get_dominant_emotion(
                        context.facial_emotion, context.voice_emotion
                    ),
                    "interest_level": context.interest_level
                },
                "phase_transition": {
                    "occurred": should_transition,
                    "new_phase": new_phase
                } if should_transition else None,
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send(json.dumps(response))
            logger.info(f"Coaching response sent for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error handling multimodal input: {e}")
            await self._send_error(websocket, f"Failed to process input: {str(e)}")
    
    async def _handle_feedback(self, websocket, data: Dict):
        """Handle user feedback on coaching responses"""
        try:
            session_id = self.connection_sessions.get(websocket)
            if not session_id:
                await self._send_error(websocket, "No active session")
                return
            
            response_id = data.get("response_id")
            effectiveness_score = data.get("effectiveness_score", 0.5)
            feedback_text = data.get("feedback", "")
            
            # Update response effectiveness
            if response_id:
                self.coaching_system.update_response_effectiveness(
                    response_id, effectiveness_score, feedback_text
                )
            
            # Update session feedback
            await self.session_manager.add_feedback(session_id, {
                "response_id": response_id,
                "effectiveness_score": effectiveness_score,
                "feedback": feedback_text,
                "timestamp": datetime.now().isoformat()
            })
            
            # Send acknowledgment
            response = {
                "type": "feedback_received",
                "session_id": session_id,
                "message": "Thank you for your feedback"
            }
            
            await websocket.send(json.dumps(response))
            logger.info(f"Feedback received for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error handling feedback: {e}")
            await self._send_error(websocket, f"Failed to process feedback: {str(e)}")
    
    async def _handle_end_session(self, websocket, data: Dict):
        """Handle session termination"""
        try:
            session_id = self.connection_sessions.get(websocket)
            if not session_id:
                await self._send_error(websocket, "No active session")
                return
            
            # End session
            await self.session_manager.end_session(session_id)
            
            # Remove connection association
            del self.connection_sessions[websocket]
            
            # Send confirmation
            response = {
                "type": "session_ended",
                "session_id": session_id,
                "message": "Session ended successfully"
            }
            
            await websocket.send(json.dumps(response))
            logger.info(f"Session ended: {session_id}")
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            await self._send_error(websocket, f"Failed to end session: {str(e)}")
    
    async def _handle_get_session_state(self, websocket, data: Dict):
        """Handle request for current session state"""
        try:
            session_id = self.connection_sessions.get(websocket)
            if not session_id:
                await self._send_error(websocket, "No active session")
                return
            
            session_state = await self.session_manager.get_session_state(session_id)
            
            response = {
                "type": "session_state",
                "session_id": session_id,
                "state": session_state
            }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            logger.error(f"Error getting session state: {e}")
            await self._send_error(websocket, f"Failed to get session state: {str(e)}")
    
    async def _handle_phase_transition(self, websocket, data: Dict):
        """Handle manual phase transition request"""
        try:
            session_id = self.connection_sessions.get(websocket)
            if not session_id:
                await self._send_error(websocket, "No active session")
                return
            
            new_phase = data.get("new_phase")
            if not new_phase:
                await self._send_error(websocket, "new_phase is required")
                return
            
            # Update session phase
            await self.session_manager.update_session_state(session_id, {
                "current_phase": new_phase,
                "phase_transition_time": datetime.now().isoformat()
            })
            
            response = {
                "type": "phase_transitioned",
                "session_id": session_id,
                "new_phase": new_phase,
                "message": f"Transitioned to {new_phase.upper()} phase"
            }
            
            await websocket.send(json.dumps(response))
            logger.info(f"Manual phase transition to {new_phase} for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error handling phase transition: {e}")
            await self._send_error(websocket, f"Failed to transition phase: {str(e)}")
    
    async def _check_phase_transition(self, context: MultimodalContext, session_state: Dict) -> bool:
        """Check if phase transition should occur"""
        # Simple heuristics for phase transition
        # You can make this more sophisticated based on your needs
        
        current_phase = session_state["current_phase"]
        conversation_turn = session_state["conversation_turn"]
        
        # Minimum turns per phase
        min_turns = {"goal": 3, "reality": 3, "options": 4, "will": 2}
        
        # Check if minimum turns completed
        if conversation_turn < min_turns.get(current_phase, 3):
            return False
        
        # Check goal clarity for goal phase
        if current_phase == "goal" and context.goal_clarity_score > 0.7:
            return True
        
        # Check interest level for engagement
        if context.interest_level > 0.7 and conversation_turn >= min_turns.get(current_phase, 3):
            return True
        
        return False
    
    async def _transition_phase(self, session_id: str, session_state: Dict) -> Optional[str]:
        """Transition to next GROW phase"""
        phase_order = ["goal", "reality", "options", "will"]
        current_phase = session_state["current_phase"]
        
        try:
            current_index = phase_order.index(current_phase)
            if current_index < len(phase_order) - 1:
                new_phase = phase_order[current_index + 1]
                
                await self.session_manager.update_session_state(session_id, {
                    "current_phase": new_phase,
                    "phase_transition_time": datetime.now().isoformat()
                })
                
                return new_phase
        except ValueError:
            logger.error(f"Invalid phase: {current_phase}")
        
        return None
    
    async def _send_error(self, websocket, error_message: str):
        """Send error message to client"""
        error_response = {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await websocket.send(json.dumps(error_response))
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")
    
    async def _cleanup_connection(self, websocket):
        """Clean up connection resources"""
        # Remove from active connections
        self.active_connections.discard(websocket)
        
        # End associated session if exists
        if websocket in self.connection_sessions:
            session_id = self.connection_sessions[websocket]
            try:
                await self.session_manager.end_session(session_id)
            except Exception as e:
                logger.error(f"Error ending session during cleanup: {e}")
            
            del self.connection_sessions[websocket]
    
    def get_active_connections_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_active_sessions_count(self) -> int:
        """Get number of active sessions"""
        return len(self.connection_sessions)

# Example usage for testing
async def main():
    """Example usage"""
    from core.coaching_rag_system import CoachingRAGSystem
    from core.session_manager import SessionManager
    
    # Initialize components
    coaching_system = CoachingRAGSystem("your_api_key")
    session_manager = SessionManager(coaching_system)
    
    # Start server
    server = RealTimeCoachingServer(coaching_system, session_manager)
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())