import chromadb
import google.generativeai as genai
from chromadb.config import Settings
import uuid
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass, asdict
import logging
from enum import Enum
from config.settings import GEMINI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GROWPhase(Enum):
    GOAL = "goal"
    REALITY = "reality" 
    OPTIONS = "options"
    WILL = "will"

class VARKType(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    READING = "reading"
    KINESTHETIC = "kinesthetic"

@dataclass
class MultimodalContext:
    """Container for all multimodal inputs and context"""
    user_id: str
    session_id: str
    timestamp: datetime
    grow_phase: GROWPhase
    utterance: str
    
    # Emotion/Sentiment data
    facial_emotion: Dict[str, float]  # {'happy': 0.8, 'sad': 0.2, etc.}
    voice_emotion: Dict[str, float]
    text_sentiment: Dict[str, float]  # {'positive': 0.7, 'negative': 0.3}
    
    # Learning style and communication patterns
    vark_type: VARKType
    vark_confidence: float
    
    # Behavioral indicators
    sarcasm_detected: bool
    sarcasm_confidence: float
    interest_level: float  # 0-1 scale
    digression_score: float
    
    # Contextual metadata
    conversation_turn: int
    previous_phase_completion: bool
    goal_clarity_score: float  # For tracking progress
    system_instruction: Optional[str] = ""   

class CoachingRAGSystem:
    """Main RAG system for multimodal AI coaching"""
    
    def __init__(self, gemini_api_key: str, chroma_persist_dir: str = "./chroma_db"):
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create collections for different data types
        self._create_collections()
        
        # Rate limiting for free API
        self.last_api_call = 0
        self.min_api_interval = 1.0  # 1 second between calls
        
    def _create_collections(self):
        """Create ChromaDB collections for different data types"""
        
        # User utterances and contexts
        self.context_collection = self.chroma_client.get_or_create_collection(
            name="coaching_contexts",
            metadata={"description": "User utterances with multimodal context"}
        )
        
        # Generated coaching responses
        self.response_collection = self.chroma_client.get_or_create_collection(
            name="coaching_responses", 
            metadata={"description": "Generated coaching responses and their effectiveness"}
        )
        
        # GROW phase templates and best practices
        self.template_collection = self.chroma_client.get_or_create_collection(
            name="grow_templates",
            metadata={"description": "GROW model templates and coaching strategies"}
        )
        
        # Initialize with some coaching templates
        self._populate_initial_templates()
    
    def _populate_initial_templates(self):
        """Populate initial coaching templates and strategies"""
        templates = [
            {
                "phase": "goal",
                "strategy": "clarifying_questions",
                "content": "Help the coachee define specific, measurable goals. Ask: What exactly do you want to achieve? By when? How will you know you've succeeded?",
                "vark_adaptations": {
                    "visual": "Use visual metaphors and ask them to visualize success",
                    "auditory": "Use verbal exploration and discuss how success would sound",
                    "reading": "Provide frameworks and written goal-setting templates", 
                    "kinesthetic": "Encourage hands-on planning and physical movement"
                }
            },
            {
                "phase": "reality",
                "strategy": "current_state_assessment",
                "content": "Explore the current situation objectively. Ask: What's happening now? What have you tried? What resources do you have?",
                "emotion_responses": {
                    "frustrated": "I can sense some frustration. Let's break down what's really happening step by step.",
                    "confused": "It sounds like there's some uncertainty here. Let's clarify the current situation together.",
                    "overwhelmed": "There seems to be a lot going on. Let's focus on the most important aspects first."
                }
            },
            {
                "phase": "options",
                "strategy": "creative_exploration",
                "content": "Generate multiple possibilities without judgment. Ask: What could you do? What if there were no constraints? Who could help?",
                "sarcasm_handling": "I notice some skepticism in your tone. Let's explore even the options that seem unlikely - sometimes they lead to breakthrough insights."
            },
            {
                "phase": "will",
                "strategy": "commitment_building",
                "content": "Solidify commitment and action steps. Ask: What will you do? When will you start? What might get in the way?",
                "low_interest_response": "I'm sensing some hesitation. What would make this feel more engaging or meaningful for you?"
            }
        ]
        
        for i, template in enumerate(templates):
            self.template_collection.add(
                documents=[template["content"]],
                metadatas=[{
                    "phase": template["phase"],
                    "strategy": template["strategy"],
                    "template_data": json.dumps(template)
                }],
                ids=[f"template_{i}"]
            )
    
    def store_context(self, context: MultimodalContext) -> str:
        """Store multimodal context in ChromaDB"""
        
        # Create document text combining utterance and context
        doc_text = f"""
        User Utterance: {context.utterance}
        GROW Phase: {context.grow_phase.value}
        VARK Type: {context.vark_type.value}
        Dominant Emotion: {self._get_dominant_emotion(context.facial_emotion, context.voice_emotion)}
        Interest Level: {context.interest_level}
        Sarcasm: {'Detected' if context.sarcasm_detected else 'None'}
        """
        
        # Prepare metadata
        metadata = {
            "user_id": context.user_id,
            "session_id": context.session_id,
            "timestamp": context.timestamp.isoformat(),
            "grow_phase": context.grow_phase.value,
            "vark_type": context.vark_type.value,
            "dominant_emotion": self._get_dominant_emotion(context.facial_emotion, context.voice_emotion),
            "interest_level": context.interest_level,
            "sarcasm_detected": context.sarcasm_detected,
            "conversation_turn": context.conversation_turn,
            "goal_clarity_score": context.goal_clarity_score,
            "context_data": json.dumps(asdict(context), default=str)
        }
        
        # Generate unique ID
        doc_id = f"{context.user_id}_{context.session_id}_{context.conversation_turn}"
        
        # Store in ChromaDB
        self.context_collection.add(
            documents=[doc_text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        logger.info(f"Stored context for user {context.user_id}, turn {context.conversation_turn}")
        return doc_id
    
    def retrieve_relevant_context(self, current_context: MultimodalContext, n_results: int = 5) -> List[Dict]:
        """Retrieve relevant past contexts and templates"""
        
        # Build query text
        query_text = f"""
        {current_context.utterance}
        Phase: {current_context.grow_phase.value}
        Emotion: {self._get_dominant_emotion(current_context.facial_emotion, current_context.voice_emotion)}
        VARK: {current_context.vark_type.value}
        """
        
        # Query for similar contexts from same user
        user_contexts = self.context_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"user_id": current_context.user_id}
        )
        
        # Query for relevant templates
        template_query = f"phase:{current_context.grow_phase.value} {current_context.vark_type.value}"
        templates = self.template_collection.query(
            query_texts=[template_query],
            n_results=3,
            where={"phase": current_context.grow_phase.value}
        )
        
        # Query for successful responses in similar contexts
        successful_responses = self.response_collection.query(
            query_texts=[query_text],
            n_results=3,
            where={"effectiveness_score": {"$gt": 0.7}}  # Only highly effective responses
        )
        
        return {
            "user_contexts": user_contexts,
            "templates": templates,
            "successful_responses": successful_responses
        }
    
    def generate_coaching_response(self, context: MultimodalContext) -> str:
        """Generate coaching response using Gemini with RAG"""
        phase = context.grow_phase
        context.system_instruction = self.get_phase_prompt(phase)

        context.digression_score = self.evaluate_digression_score(context.utterance, phase)
        logger.info(f"Calculated digression score: {context.digression_score}")

        if phase == GROWPhase.GOAL:
            context.goal_clarity_score = self.evaluate_goal_clarity(context.utterance)
            
        # Rate limiting for free API
        self._rate_limit()
        
        # Retrieve relevant information
        retrieved_data = self.retrieve_relevant_context(context)
        
        # Build context-aware prompt
        prompt = self._build_coaching_prompt(context, retrieved_data)
        
        try:
            # Generate response with Gemini
            response = self.gemini_model.generate_content(prompt)
            generated_response = response.text
            
            # Store the response
            self._store_coaching_response(context, generated_response, retrieved_data)
            
            return generated_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback to template-based response
            return self._fallback_response(context, retrieved_data)
    
    def _build_coaching_prompt(self, context: MultimodalContext, retrieved_data: Dict) -> str:
        """Build comprehensive prompt for Gemini"""
        
        # Extract key information
        dominant_emotion = self._get_dominant_emotion(context.facial_emotion, context.voice_emotion)
        
        prompt = f"""
        {context.system_instruction}
You are an expert AI coach using the GROW model. Generate a personalized coaching response based on the following context:

CURRENT SITUATION:
- GROW Phase: {context.grow_phase.value.upper()}
- User's Statement: "{context.utterance}"
- Learning Style (VARK): {context.vark_type.value}
- Dominant Emotion: {dominant_emotion}
- Interest Level: {context.interest_level}/1.0
- Sarcasm Detected: {context.sarcasm_detected}
- Conversation Turn: {context.conversation_turn}
- Goal Clarity Score: {context.goal_clarity_score}/1.0
- Digression Score: {context.digression_score}/1.0

RELEVANT COACHING TEMPLATES:
{self._format_templates(retrieved_data.get('templates', {}))}

SIMILAR PAST INTERACTIONS:
{self._format_past_contexts(retrieved_data.get('user_contexts', {}))}

COACHING GUIDELINES:
1. Adapt your response to the user's VARK learning style
2. Acknowledge and respond appropriately to their emotional state
3. Stay focused on the current GROW phase while building on previous phases
4. If sarcasm is detected, address it constructively
5. If interest is low, use engagement techniques
6. Keep responses concise but impactful (2-3 sentences max)
7. Ask one powerful coaching question to move the conversation forward
8. **If the digression_score is high (above 0.5), your primary task is to steer the user back.** Briefly acknowledge their comment (e.g., "I see," or "Thanks for sharing,"), then immediately pivot back to the coaching topic.
9. **CRITICAL: Do NOT engage with the user's off-topic subject.** Do not ask questions about it or discuss it further. Your sole focus must be returning to the coaching phase.

Generate a coaching response that is:
- Empathetic and personalized
- Aligned with GROW methodology
- Adapted to their learning style
- Appropriate for their emotional state
- Actionable and engaging
"""
        
        return prompt
    
    def _store_coaching_response(self, context: MultimodalContext, response: str, retrieved_data: Dict):
        """Store generated coaching response for future retrieval"""
        
        doc_text = f"""
        Coaching Response: {response}
        Context: {context.utterance}
        Phase: {context.grow_phase.value}
        User Emotion: {self._get_dominant_emotion(context.facial_emotion, context.voice_emotion)}
        VARK Type: {context.vark_type.value}
        """
        
        metadata = {
            "user_id": context.user_id,
            "session_id": context.session_id,
            "timestamp": datetime.now().isoformat(),
            "grow_phase": context.grow_phase.value,
            "response_type": "generated",
            "context_summary": json.dumps({
                "utterance": context.utterance,
                "emotion": self._get_dominant_emotion(context.facial_emotion, context.voice_emotion),
                "vark": context.vark_type.value,
                "interest": context.interest_level
            }),
            "effectiveness_score": 0.5  # Default, to be updated based on feedback
        }
        
        response_id = f"response_{context.user_id}_{context.session_id}_{context.conversation_turn}"
        
        self.response_collection.add(
            documents=[doc_text],
            metadatas=[metadata],
            ids=[response_id]
        )
    
    def update_response_effectiveness(self, response_id: str, effectiveness_score: float, feedback: str = ""):
        """Update effectiveness score based on user feedback"""
        
        # Get current metadata
        result = self.response_collection.get(ids=[response_id], include=["metadatas"])
        if result["ids"]:
            metadata = result["metadatas"][0]
            metadata["effectiveness_score"] = effectiveness_score
            metadata["feedback"] = feedback
            metadata["updated_at"] = datetime.now().isoformat()
            
            # Update the document
            self.response_collection.update(
                ids=[response_id],
                metadatas=[metadata]
            )
    
    def export_training_data(self, output_path: str):
        """Export data for training custom models"""
        
        # Get all contexts and responses
        all_contexts = self.context_collection.get(include=["documents", "metadatas"])
        all_responses = self.response_collection.get(include=["documents", "metadatas"])
        
        training_data = {
            "contexts": [
                {
                    "id": all_contexts["ids"][i],
                    "document": all_contexts["documents"][i],
                    "metadata": all_contexts["metadatas"][i]
                }
                for i in range(len(all_contexts["ids"]))
            ],
            "responses": [
                {
                    "id": all_responses["ids"][i], 
                    "document": all_responses["documents"][i],
                    "metadata": all_responses["metadatas"][i]
                }
                for i in range(len(all_responses["ids"]))
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Training data exported to {output_path}")
        return len(training_data["contexts"]), len(training_data["responses"])
    
    def _get_dominant_emotion(self, facial_emotions: Dict, voice_emotions: Dict) -> str:
        """Get the dominant emotion from multimodal inputs"""
        
        # Combine and weight emotions (you can adjust weights based on your needs)
        combined_emotions = {}
        
        # Weight facial emotions (visual cues)
        for emotion, score in facial_emotions.items():
            combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score * 0.6
        
        # Weight voice emotions (audio cues)  
        for emotion, score in voice_emotions.items():
            combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score * 0.4
        
        # Return emotion with highest combined score
        if combined_emotions:
            return max(combined_emotions, key=combined_emotions.get)
        return "neutral"
    
    def _rate_limit(self):
        """Simple rate limiting for free API"""
        current_time = time.time()
        if current_time - self.last_api_call < self.min_api_interval:
            time.sleep(self.min_api_interval - (current_time - self.last_api_call))
        self.last_api_call = time.time()
    
    def _format_templates(self, templates_data: Dict) -> str:
        """Format template data for prompt"""
        if not templates_data or not templates_data.get("documents"):
            return "No relevant templates found."
        
        formatted = []
        for i, doc in enumerate(templates_data["documents"]):
            metadata = templates_data["metadatas"][i] if i < len(templates_data.get("metadatas", [])) else {}
            formatted.append(f"Template: {doc}")
        
        return "\n".join(formatted[:3])  # Limit to top 3
    
    def _format_past_contexts(self, contexts_data: Dict) -> str:
        """Format past context data for prompt"""
        if not contexts_data or not contexts_data.get("documents"):
            return "No relevant past interactions found."
        
        formatted = []
        for i, doc in enumerate(contexts_data["documents"]):
            formatted.append(f"Past Interaction: {doc}")
        
        return "\n".join(formatted[:3])  # Limit to top 3
    
    def _fallback_response(self, context: MultimodalContext, retrieved_data: Dict) -> str:
        """Fallback response when API fails"""
        
        # Simple template-based fallback
        phase_responses = {
            GROWPhase.GOAL: "What specifically would you like to achieve? Let's make your goal clear and actionable.",
            GROWPhase.REALITY: "Tell me more about your current situation. What's working and what isn't?", 
            GROWPhase.OPTIONS: "What are some different ways you could approach this? Let's explore your options.",
            GROWPhase.WILL: "What's your next step? When will you take action?"
        }
        
        base_response = phase_responses.get(context.grow_phase, "How can I help you move forward?")
        
        # Adapt for low interest
        if context.interest_level < 0.5:
            base_response += " What would make this more engaging for you?"
        
        return base_response
    
#################################################################

    def query_llm(self, prompt: str) -> str:
        try:
            response = self.gemini.generate_text(prompt)
            return response.text if hasattr(response, 'text') else response
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return "no"
    
    def get_phase_prompt(self, phase: GROWPhase) -> str:
        prompts = {
            GROWPhase.GOAL: "Help the user clarify their coaching goal. Ask reflective questions to sharpen their focus.",
            GROWPhase.REALITY: "Explore the user's current situation with empathy. Ask about obstacles, support systems, and what’s working.",
            GROWPhase.OPTIONS: "Help the user brainstorm possible ways forward. Offer suggestions, but empower them to generate ideas.",
            GROWPhase.WILL: "Guide the user to commit to specific actions. Ask about motivation and accountability."
        }
        return prompts.get(phase, "")
    
    def evaluate_goal_clarity(self, user_input: str) -> float:
        """Use LLM to estimate how clear the user's coaching goal is."""
        prompt = f"""Rate the clarity of this coaching goal from 0.0 (very vague) to 1.0 (very clear):
    \"{user_input}\"

    Respond with just a number (e.g., 0.8)."""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            score = float(response.text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"Goal clarity evaluation failed: {e}")
            return 0.3

    # ADDED: New function to evaluate digression
    def evaluate_digression_score(self, user_input: str, phase: GROWPhase) -> float:
        """Use LLM to score how much the user is digressing from the current GROW phase."""
        
        # Get a description of what the current phase is about
        phase_description = self.get_phase_prompt(phase)

        prompt = f"""You are an expert in conversation analysis for a coaching session. The user is currently in the '{phase.value.upper()}' phase of the GROW model. The goal of this phase is to: "{phase_description}"

Based on the user's latest statement, please rate on a scale from 0.0 (perfectly on-topic) to 1.0 (completely off-topic) how much they are digressing from this phase.

User's Statement: "{user_input}"

Respond with just a number (e.g., 0.1 or 0.9)."""
        
        self._rate_limit() # Respect API rate limits
        try:
            response = self.gemini_model.generate_content(prompt)
            score = float(response.text.strip())
            # Ensure the score is within the 0.0 to 1.0 range
            return max(0.0, min(1.0, score))
        except (ValueError, AttributeError, Exception) as e:
            logger.error(f"Digression score evaluation failed: {e}")
            # Default to a low score if evaluation fails
            return 0.1

# Usage Example
def main():
    """Example usage of the coaching system"""
    
    # Initialize system
    coaching_system = CoachingRAGSystem(
        gemini_api_key=GEMINI_API_KEY,
        chroma_persist_dir="./coaching_chroma_db"
    )
    
    # Example multimodal context
    context = MultimodalContext(
        user_id="user_123",
        session_id="session_456", 
        timestamp=datetime.now(),
        grow_phase=GROWPhase.GOAL,
        utterance="I want to get better at public speaking but I'm not sure where to start",
        
        facial_emotion={"nervous": 0.7, "hopeful": 0.3},
        voice_emotion={"anxious": 0.6, "determined": 0.4},
        text_sentiment={"positive": 0.4, "negative": 0.6},
        
        vark_type=VARKType.VISUAL,
        vark_confidence=0.8,
        
        sarcasm_detected=False,
        sarcasm_confidence=0.1,
        interest_level=0.7,
        digression_detected=0.1,
        
        conversation_turn=1,
        previous_phase_completion=False,
        goal_clarity_score=0.3
    )
    
    # Store context
    context_id = coaching_system.store_context(context)
    print(f"Stored context: {context_id}")
    
    # Generate coaching response
    response = coaching_system.generate_coaching_response(context)
    print(f"Coaching Response: {response}")
    
    # Export training data
    num_contexts, num_responses = coaching_system.export_training_data("training_data.json")
    print(f"Exported {num_contexts} contexts and {num_responses} responses")

if __name__ == "__main__":
    main()