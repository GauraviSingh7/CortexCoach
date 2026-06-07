"""
Interest Level Detection Model Interface
=======================================

This module provides inference for detecting interest levels in conversation
based on text analysis and audio features.
"""

import numpy as np
import logging
from typing import Optional, Dict, List
from pathlib import Path
import joblib
import torch
import re
from collections import Counter

logger = logging.getLogger(__name__)

class InterestDetectionModel:
    """
    Interest level detection model that analyzes text patterns and audio features
    to determine engagement and interest levels.
    """
    
    # High interest indicators
    HIGH_INTEREST_WORDS = [
        'excited', 'amazing', 'fantastic', 'wonderful', 'great', 'awesome',
        'love', 'fascinating', 'interesting', 'curious', 'eager', 'motivated',
        'passionate', 'thrilled', 'impressed', 'wow', 'incredible', 'brilliant'
    ]
    
    # Low interest indicators
    LOW_INTEREST_WORDS = [
        'boring', 'tired', 'exhausted', 'uninterested', 'whatever', 'meh',
        'okay', 'fine', 'sure', 'maybe', 'doubt', 'uncertain', 'confused',
        'lost', 'overwhelmed', 'frustrated', 'annoyed'
    ]
    
    # Engagement patterns
    ENGAGEMENT_PATTERNS = {
        'questions': r'\?',
        'exclamations': r'!',
        'elaboration': r'\b(because|since|so|therefore|thus|hence)\b',
        'examples': r'\b(for example|for instance|such as|like)\b',
        'agreement': r'\b(yes|yeah|absolutely|exactly|definitely|right)\b',
        'disagreement': r'\b(no|but|however|although|though)\b'
    }
    
    def __init__(self, model_path: Path):
        """
        Initialize the interest detection model.
        
        Args:
            model_path: Path to the model directory
        """
        self.model_path = Path(model_path)
        self.text_model = None
        self.audio_model = None
        self.vectorizer = None
        self.scaler = None
        
        # Load model components
        self._load_models()
        
    def _load_models(self):
        """Load all model components"""
        try:
            pipeline_path = self.model_path / "engagement_pipeline.pkl"
            if pipeline_path.exists():
                self.text_model = joblib.load(pipeline_path)
                logger.info("Interest detection pipeline loaded successfully")
            else:
                logger.warning("no engagement_pipeline.pkl found")
        except Exception as e:
            logger.error(f"Error loading interest detection pipeline :{e}")
    
    def predict(self, text: str, audio_data: Optional[bytes] = None) -> float:
        """
        Predict interest level from text and/or audio data.
        
        Args:
            text: Input text to analyze
            audio_data: Raw audio bytes (optional)
            
        Returns:
            Interest score between 0.0 (low) and 1.0 (high)
        """
        try:
            # Get text-based interest score
            text_score = self._predict_from_text(text)
            
            # Get audio-based interest score if available
            if audio_data and self.audio_model:
                audio_score = self._predict_from_audio(audio_data)
                # Combine text and audio scores (weighted average)
                combined_score = 0.7 * text_score + 0.3 * audio_score
            else:
                combined_score = text_score
            
            # Ensure score is in valid range
            return max(0.0, min(1.0, combined_score))
            
        except Exception as e:
            logger.error(f"Error in interest prediction: {e}")
            return 0.5  # Neutral fallback
    
    def _predict_from_text(self, text: str) -> float:
        """Predict interest level from text"""
        try:
            if self.text_model:
                probs = None
                if hasattr(self.text_model, "predict_proba"):
                    probs = self.text_model.predict_proba([text])[0]
                    return float(probs[1] if len(probs)>1 else probs[0])
                else:
                    return float(self.text_model.predict([text]))[0]
            else:
                return self._predict_rule_based_text(text)
        except Exception as e:
            logger.error(f"text interest prediction error: {e}")
            return self._predict_rule_based_text(text)
    
    def _predict_ml_text(self, text: str) -> float:
        """Predict using trained ML text model"""
        try:
            # Vectorize text
            text_vector = self.vectorizer.transform([text])
            
            # Scale if scaler is available
            if self.scaler:
                text_vector = self.scaler.transform(text_vector)
            
            # Get prediction
            if hasattr(self.text_model, 'predict_proba'):
                # Binary classifier with probabilities
                probs = self.text_model.predict_proba(text_vector)[0]
                # Assume classes are [low_interest, high_interest]
                interest_score = probs[1] if len(probs) > 1 else probs[0]
            else:
                # Regression model
                interest_score = self.text_model.predict(text_vector)[0]
            
            return float(interest_score)
            
        except Exception as e:
            logger.error(f"ML text interest prediction error: {e}")
            return self._predict_rule_based_text(text)
    
    def _predict_rule_based_text(self, text: str) -> float:
        """Predict using rule-based text analysis"""
        try:
            text_lower = text.lower().strip()
            
            if not text_lower:
                return 0.3  # Low score for empty text
            
            # Base score
            interest_score = 0.5
            
            # Analyze word-level indicators
            words = re.findall(r'\b\w+\b', text_lower)
            word_count = len(words)
            
            # High interest word indicators
            high_interest_count = sum(1 for word in words if word in self.HIGH_INTEREST_WORDS)
            if high_interest_count > 0:
                interest_score += min(0.3, high_interest_count * 0.1)
            
            # Low interest word indicators
            low_interest_count = sum(1 for word in words if word in self.LOW_INTEREST_WORDS)
            if low_interest_count > 0:
                interest_score -= min(0.3, low_interest_count * 0.1)
            
            # Analyze engagement patterns
            engagement_score = self._analyze_engagement_patterns(text_lower)
            interest_score += engagement_score * 0.2
            
            # Text length and complexity indicators
            length_score = self._analyze_text_complexity(text_lower, word_count)
            interest_score += length_score * 0.1
            
            # Emotional intensity indicators
            emotional_score = self._analyze_emotional_intensity(text_lower)
            interest_score += emotional_score * 0.15
            
            # Response time simulation (placeholder - you might have actual timing data)
            response_speed_score = self._simulate_response_speed_score(text_lower)
            interest_score += response_speed_score * 0.1
            
            # Normalize to [0, 1] range
            return max(0.0, min(1.0, interest_score))
            
        except Exception as e:
            logger.error(f"Rule-based text interest prediction error: {e}")
            return 0.5
    
    def _analyze_engagement_patterns(self, text: str) -> float:
        """Analyze patterns that indicate engagement"""
        engagement_score = 0.0
        
        for pattern_name, pattern in self.ENGAGEMENT_PATTERNS.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            
            if pattern_name == 'questions':
                # Questions show curiosity and engagement
                engagement_score += min(0.5, matches * 0.2)
            elif pattern_name == 'exclamations':
                # Exclamations show enthusiasm
                engagement_score += min(0.3, matches * 0.15)
            elif pattern_name == 'elaboration':
                # Elaboration shows deeper thinking
                engagement_score += min(0.4, matches * 0.1)
            elif pattern_name == 'examples':
                # Examples show active participation
                engagement_score += min(0.3, matches * 0.1)
            elif pattern_name == 'agreement':
                # Agreement shows engagement (moderate boost)
                engagement_score += min(0.2, matches * 0.05)
            elif pattern_name == 'disagreement':
                # Thoughtful disagreement also shows engagement
                engagement_score += min(0.2, matches * 0.05)
        
        return engagement_score
    
    def _analyze_text_complexity(self, text: str, word_count: int) -> float:
        """Analyze text complexity as interest indicator"""
        complexity_score = 0.0
        
        if word_count == 0:
            return -0.5  # Very low for no content
        
        # Length indicators
        if word_count > 20:
            complexity_score += 0.3  # Longer responses show more engagement
        elif word_count > 10:
            complexity_score += 0.2
        elif word_count < 3:
            complexity_score -= 0.2  # Very short might indicate low interest
        
        # Sentence complexity
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = word_count / max(1, len(sentences))
        
        if avg_sentence_length > 8:
            complexity_score += 0.2  # Complex sentences show thoughtfulness
        elif avg_sentence_length < 3:
            complexity_score -= 0.1  # Very short sentences might show disengagement
        
        # Vocabulary diversity
        unique_words = len(set(re.findall(r'\b\w+\b', text.lower())))
        if word_count > 0:
            diversity_ratio = unique_words / word_count
            if diversity_ratio > 0.7:
                complexity_score += 0.2  # High vocabulary diversity
        
        return complexity_score
    
    def _analyze_emotional_intensity(self, text: str) -> float:
        """Analyze emotional intensity in text"""
        intensity_score = 0.0
        
        # Intensity indicators
        intensity_patterns = [
            (r'\b(very|really|extremely|incredibly|absolutely|totally)\b', 0.1),
            (r'\b(so|such)\b', 0.05),
            (r'[A-Z]{2,}', 0.1),  # ALL CAPS words
            (r'(.)\1{2,}', 0.1),  # Repeated characters (sooo, yesss)
        ]
        
        for pattern, weight in intensity_patterns:
            matches = len(re.findall(pattern, text))
            intensity_score += min(0.3, matches * weight)
        
        return intensity_score
    
    def _simulate_response_speed_score(self, text: str) -> float:
        """
        Simulate response speed scoring.
        In a real implementation, you would use actual response timing data.
        """
        # Placeholder: assume longer, more complex responses took more time
        # and might indicate higher engagement
        word_count = len(re.findall(r'\b\w+\b', text))
        
        if word_count > 15:
            return 0.2  # Thoughtful, longer response
        elif word_count > 5:
            return 0.1  # Moderate response
        else:
            return -0.1  # Very brief might indicate low interest
    
    def _predict_from_audio(self, audio_data: bytes) -> float:
        """Predict interest level from audio features"""
        try:
            # Extract audio features
            audio_features = self._extract_audio_features(audio_data)
            
            if audio_features is None:
                logger.warning("Could not extract audio features")
                return 0.5
            
            # Scale features if scaler is available
            if self.scaler:
                audio_features = self.scaler.transform([audio_features])
            else:
                audio_features = np.array([audio_features])
            
            # Get predictions from audio model
            if isinstance(self.audio_model, torch.nn.Module):
                # PyTorch model
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(audio_features)
                    output = self.audio_model(features_tensor)
                    interest_score = torch.sigmoid(output).item()  # Assuming binary output
            else:
                # Sklearn model
                if hasattr(self.audio_model, 'predict_proba'):
                    probs = self.audio_model.predict_proba(audio_features)[0]
                    interest_score = probs[1] if len(probs) > 1 else probs[0]
                else:
                    interest_score = self.audio_model.predict(audio_features)[0]
            
            return float(interest_score)
            
        except Exception as e:
            logger.error(f"Audio interest prediction error: {e}")
            return 0.5
    
    def _extract_audio_features(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Extract audio features that correlate with interest/engagement.
        This is a placeholder - implement based on your audio processing pipeline.
        """
        try:
            # Placeholder implementation
            # In a real implementation, you would extract features like:
            # - Pitch variation (more variation often indicates engagement)
            # - Speaking rate (faster might indicate excitement)
            # - Volume/energy (higher energy often indicates interest)
            # - Pause patterns (fewer, shorter pauses might indicate engagement)
            # - Voice quality features
            
            logger.warning("Using dummy audio features - implement actual feature extraction")
            return np.random.rand(20)  # Assuming 20-dimensional features
            
        except Exception as e:
            logger.error(f"Audio feature extraction error: {e}")
            return None
    
    def analyze_conversation_trend(self, conversation_history: List[Dict]) -> Dict[str, float]:
        """
        Analyze interest trends across conversation history.
        
        Args:
            conversation_history: List of conversation turns with timestamps and scores
            
        Returns:
            Dictionary with trend analysis
        """
        if not conversation_history:
            return {"trend": "neutral", "average": 0.5, "volatility": 0.0}
        
        try:
            scores = [turn.get('interest_score', 0.5) for turn in conversation_history]
            
            # Calculate basic statistics
            avg_score = np.mean(scores)
            volatility = np.std(scores) if len(scores) > 1 else 0.0
            
            # Determine trend
            if len(scores) >= 3:
                recent_avg = np.mean(scores[-3:])  # Last 3 scores
                early_avg = np.mean(scores[:3])    # First 3 scores
                
                if recent_avg > early_avg + 0.1:
                    trend = "increasing"
                elif recent_avg < early_avg - 0.1:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            # Peak and valley detection
            peak_score = max(scores)
            valley_score = min(scores)
            
            return {
                "trend": trend,
                "average": float(avg_score),
                "volatility": float(volatility),
                "peak": float(peak_score),
                "valley": float(valley_score),
                "latest": float(scores[-1]) if scores else 0.5,
                "data_points": len(scores)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conversation trend: {e}")
            return {"trend": "error", "average": 0.5, "volatility": 0.0}
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        return {
            "text_model": "loaded" if self.text_model else "not available",
            "audio_model": "loaded" if self.audio_model else "not available",
            "vectorizer": "loaded" if self.vectorizer else "not available",
            "approach": "ML + rule-based" if self.text_model else "rule-based only",
            "high_interest_indicators": len(self.HIGH_INTEREST_WORDS),
            "low_interest_indicators": len(self.LOW_INTEREST_WORDS),
            "engagement_patterns": len(self.ENGAGEMENT_PATTERNS),
            "model_path": str(self.model_path)
        }
    

class EngagementPredictor(InterestDetectionModel):
    "Wrapper so old pkl models looking for EngagementPredictor still work"
    pass