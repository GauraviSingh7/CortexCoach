from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime
from core.coaching_rag_system import MultimodalContext, GROWPhase, VARKType
from models.inference_pipeline import MultimodalInferencePipeline
import logging

logger = logging.getLogger(__name__)

class MultimodalContextProcessor:
    """Integrates model outputs with the existing coaching system"""
    
    def __init__(self, model_paths: Dict[str, str]):
        self.pipeline = MultimodalInferencePipeline(model_paths)
        self.interest_tracker = InterestLevelCalculator()
    
    def process_facial_emotion(self, image: np.ndarray) -> Dict[str, float]:
        """Pass image to the underlying pipeline's facial emotion processor"""
        return self.pipeline.process_facial_emotion(image)
    
    def create_multimodal_context(
        self,
        user_id: str,
        session_id: str,
        utterance: str,
        image: Optional[np.ndarray] = None,
        grow_phase: GROWPhase = GROWPhase.GOAL,
        conversation_turn: int = 1,
        goal_clarity_score: float = 0.5
    ) -> MultimodalContext:
        """
        Create a complete MultimodalContext from raw inputs
        """
        
        # Process inputs through models
        model_outputs = self.inference_pipeline.process_multimodal_input(
            text=utterance,
            image=image
        )
        
        # Calculate interest level from multimodal cues
        interest_level = self.interest_tracker.calculate_interest_level(
            facial_emotion=model_outputs['facial_emotion'],
            text=utterance,
            sarcasm_detected=model_outputs['sarcasm_detected']
        )
        
        # Map VAK string to enum
        vark_mapping = {
            'visual': VARKType.VISUAL,
            'auditory': VARKType.AUDITORY,
            'reading': VARKType.READING,
            'kinesthetic': VARKType.KINESTHETIC
        }
        
        vark_type = vark_mapping.get(model_outputs['vak_type'], VARKType.VISUAL)
        
        # Create context object
        context = MultimodalContext(
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            grow_phase=grow_phase,
            utterance=utterance,
            
            # Emotion data
            facial_emotion=model_outputs['facial_emotion'],
            voice_emotion={},  # Placeholder - add voice emotion model if available
            text_sentiment=self._extract_text_sentiment(model_outputs['facial_emotion']),
            
            # Learning style
            vark_type=vark_type,
            vark_confidence=model_outputs['vak_confidence'],
            
            # Behavioral indicators
            sarcasm_detected=model_outputs['sarcasm_detected'],
            sarcasm_confidence=model_outputs['sarcasm_confidence'],
            interest_level=interest_level,
            digression_detected=False,  # Placeholder for future model
            
            # Session context
            conversation_turn=conversation_turn,
            previous_phase_completion=False,
            goal_clarity_score=goal_clarity_score
        )
        
        return context
    
    def _extract_text_sentiment(self, facial_emotion: Dict[str, float]) -> Dict[str, float]:
        """Convert facial emotions to text sentiment (placeholder)"""
        positive_emotions = ['happy', 'surprise']
        negative_emotions = ['angry', 'sad', 'fear', 'disgust']
        
        positive_score = sum(facial_emotion.get(emotion, 0) for emotion in positive_emotions)
        negative_score = sum(facial_emotion.get(emotion, 0) for emotion in negative_emotions)
        neutral_score = facial_emotion.get('neutral', 0)
        
        total = positive_score + negative_score + neutral_score
        if total > 0:
            return {
                'positive': positive_score / total,
                'negative': negative_score / total,
                'neutral': neutral_score / total
            }
        
        return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.33}
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Run text through VAK and sarcasm models only"""
        results = {
            'vak_type': 'visual',
            'vak_confidence': 0.0,
            'sarcasm_detected': False,
            'sarcasm_confidence': 0.0
        }

        # ✅ Convert list of strings to a single string if needed
        if isinstance(text, list):
            text = ' '.join(text)

        if not text:
            return results

        if hasattr(self.pipeline, "models"):
            # Sarcasm
            sarcasm_model = self.pipeline.models.get("sarcasm")
            if sarcasm_model:
                try:
                    is_sarcastic, conf = sarcasm_model.predict(text)
                    results['sarcasm_detected'] = is_sarcastic
                    results['sarcasm_confidence'] = conf
                except Exception as e:
                    logger.warning(f"Sarcasm prediction failed: {e}")

            # VAK
            vak_model = self.pipeline.models.get("vak")
            if vak_model:
                try:
                    vak, conf = vak_model.predict(text)
                    results['vak_type'] = vak
                    results['vak_confidence'] = conf
                except Exception as e:
                    logger.warning(f"VAK prediction failed: {e}")

        return results


class InterestLevelCalculator:
    """Calculate user interest level from multimodal cues"""
    
    def calculate_interest_level(
        self,
        facial_emotion: Dict[str, float],
        text: str,
        sarcasm_detected: bool = False
    ) -> float:
        """
        Calculate interest level (0-1) from available cues
        """
        
        # Base interest from facial emotions
        engaged_emotions = ['happy', 'surprise']
        disengaged_emotions = ['bored', 'sad', 'neutral']  # Add 'bored' if your model has it
        
        engagement_score = sum(facial_emotion.get(emotion, 0) for emotion in engaged_emotions)
        disengagement_score = sum(facial_emotion.get(emotion, 0) for emotion in disengaged_emotions)
        
        # Text-based indicators
        text_engagement = self._calculate_text_engagement(text)
        
        # Sarcasm penalty
        sarcasm_penalty = 0.2 if sarcasm_detected else 0.0
        
        # Combine scores
        base_score = (engagement_score - disengagement_score + text_engagement) / 2
        final_score = max(0.0, min(1.0, base_score - sarcasm_penalty))
        
        return final_score
    

    def _calculate_text_engagement(self, text: str) -> float:
        """Calculate engagement from text characteristics"""
        if not text:
            return 0.5
        
        # Positive indicators
        engagement_words = ['excited', 'interested', 'want', 'need', 'help', 'learn']
        disengagement_words = ['boring', 'whatever', 'don\'t care', 'waste']
        
        text_lower = text.lower()
        engagement_count = sum(word in text_lower for word in engagement_words)
        disengagement_count = sum(word in text_lower for word in disengagement_words)
        
        # Length and punctuation as engagement indicators
        length_score = min(len(text) / 100, 1.0)  # Longer responses often indicate engagement
        question_score = 0.1 if '?' in text else 0.0
        
        return (engagement_count - disengagement_count + length_score + question_score) / 4