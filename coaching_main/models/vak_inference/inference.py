"""
VAK Learning Style Inference Model Interface
============================================

This module provides inference for Visual, Auditory, and Kinesthetic learning style preferences
based on text analysis of conversation patterns and language use.
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from pathlib import Path
import joblib
import torch
import re
from collections import Counter

logger = logging.getLogger(__name__)

class VAKInferenceModel:
    """
    VAK (Visual, Auditory, Kinesthetic) learning style inference model.
    Analyzes text patterns to determine learning style preferences.
    """
    
    # Keywords and patterns associated with each learning style
    VISUAL_KEYWORDS = [
        'see', 'look', 'view', 'picture', 'image', 'visualize', 'imagine', 'show', 'display',
        'appear', 'clear', 'bright', 'colorful', 'focus', 'perspective', 'observe', 'watch',
        'diagram', 'chart', 'graph', 'map', 'sketch', 'draw', 'design', 'pattern'
    ]
    
    AUDITORY_KEYWORDS = [
        'hear', 'listen', 'sound', 'voice', 'speak', 'talk', 'say', 'tell', 'discuss',
        'explain', 'describe', 'mention', 'noise', 'quiet', 'loud', 'tone', 'rhythm',
        'music', 'verbal', 'oral', 'conversation', 'dialogue', 'debate', 'lecture'
    ]
    
    KINESTHETIC_KEYWORDS = [
        'feel', 'touch', 'move', 'action', 'do', 'try', 'practice', 'experience', 'hands-on',
        'physical', 'active', 'exercise', 'walk', 'run', 'build', 'make', 'create',
        'handle', 'manipulate', 'explore', 'experiment', 'concrete', 'tangible', 'solid'
    ]
    
    def __init__(self, model_path: Path):
        """
        Initialize the VAK inference model.
        
        Args:
            model_path: Path to the model directory
        """
        self.model_path = Path(model_path)
        self.ml_model = None
        self.vectorizer = None
        self.scaler = None
        
        # Load model components
        self._load_models()
        
    def _load_models(self):
        """Load all model components"""
        try:
            # Load ML model if available
            ml_model_path = self.model_path / "vak_classifier.pkl"
            if ml_model_path.exists():
                self.ml_model = joblib.load(ml_model_path)
                logger.info("VAK ML classifier loaded successfully")
            
            # Load vectorizer
            vectorizer_path = self.model_path / "vectorizer.pkl"
            if vectorizer_path.exists():
                self.vectorizer = joblib.load(vectorizer_path)
                logger.info("VAK vectorizer loaded successfully")
            
            # Load scaler
            scaler_path = self.model_path / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("VAK scaler loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading VAK models: {e}")
            # Continue with rule-based approach if models fail to load
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict VAK learning style preferences from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with visual, auditory, kinesthetic scores (sum to 1.0)
        """
        try:
            # Use ML model if available, otherwise fall back to rule-based
            if self.ml_model and self.vectorizer:
                return self._predict_ml(text)
            else:
                return self._predict_rule_based(text)
                
        except Exception as e:
            logger.error(f"Error in VAK prediction: {e}")
            # Return balanced fallback
            return {"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34}
    
    def _predict_ml(self, text: str) -> Dict[str, float]:
        """Predict using trained ML model"""
        try:
            # Vectorize the text
            text_features = self.vectorizer.transform([text])
            
            # Scale features if scaler is available
            if self.scaler:
                text_features = self.scaler.transform(text_features)
            
            # Get predictions
            if hasattr(self.ml_model, 'predict_proba'):
                # Multi-class classifier returning probabilities
                probs = self.ml_model.predict_proba(text_features)[0]
                return {
                    "visual": float(probs[0] if len(probs) > 0 else 0.33),
                    "auditory": float(probs[1] if len(probs) > 1 else 0.33),
                    "kinesthetic": float(probs[2] if len(probs) > 2 else 0.34)
                }
            else:
                # Single prediction - convert to probabilities
                prediction = self.ml_model.predict(text_features)[0]
                vak_scores = {"visual": 0.0, "auditory": 0.0, "kinesthetic": 0.0}
                if prediction == 0:
                    vak_scores["visual"] = 1.0
                elif prediction == 1:
                    vak_scores["auditory"] = 1.0
                else:
                    vak_scores["kinesthetic"] = 1.0
                return vak_scores
                
        except Exception as e:
            logger.error(f"ML VAK prediction error: {e}")
            return self._predict_rule_based(text)
    
    def _predict_rule_based(self, text: str) -> Dict[str, float]:
        """Predict using rule-based keyword analysis"""
        try:
            # Preprocess text
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            # Count keyword matches for each style
            visual_count = sum(1 for word in words if word in self.VISUAL_KEYWORDS)
            auditory_count = sum(1 for word in words if word in self.AUDITORY_KEYWORDS)
            kinesthetic_count = sum(1 for word in words if word in self.KINESTHETIC_KEYWORDS)
            
            # Add pattern-based features
            visual_patterns = self._extract_visual_patterns(text_lower)
            auditory_patterns = self._extract_auditory_patterns(text_lower)
            kinesthetic_patterns = self._extract_kinesthetic_patterns(text_lower)
            
            # Combine counts with pattern scores
            visual_score = visual_count + visual_patterns
            auditory_score = auditory_count + auditory_patterns
            kinesthetic_score = kinesthetic_count + kinesthetic_patterns
            
            # Apply contextual weighting
            visual_score *= self._get_context_weight('visual', text_lower)
            auditory_score *= self._get_context_weight('auditory', text_lower)
            kinesthetic_score *= self._get_context_weight('kinesthetic', text_lower)
            
            # Normalize to probabilities
            total_score = visual_score + auditory_score + kinesthetic_score
            
            if total_score == 0:
                # No indicators found - return balanced distribution
                return {"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34}
            
            return {
                "visual": visual_score / total_score,
                "auditory": auditory_score / total_score,
                "kinesthetic": kinesthetic_score / total_score
            }
            
        except Exception as e:
            logger.error(f"Rule-based VAK prediction error: {e}")
            return {"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34}
    
    def _extract_visual_patterns(self, text: str) -> float:
        """Extract visual learning patterns from text"""
        score = 0.0
        
        # Visual metaphors and expressions
        visual_phrases = [
            r'i see what you mean', r'looks like', r'appears to be',
            r'from my perspective', r'picture this', r'envision',
            r'it\'s clear that', r'shed light on', r'bright idea',
            r'colorful description', r'paint a picture', r'illustrate'
        ]
        
        for phrase in visual_phrases:
            if re.search(phrase, text):
                score += 0.5
        
        # References to visual media
        visual_media = [
            r'chart', r'graph', r'diagram', r'map', r'image',
            r'video', r'screen', r'display', r'presentation'
        ]
        
        for media in visual_media:
            score += len(re.findall(media, text)) * 0.3
        
        return score
    
    def _extract_auditory_patterns(self, text: str) -> float:
        """Extract auditory learning patterns from text"""
        score = 0.0
        
        # Auditory expressions
        auditory_phrases = [
            r'i hear you', r'sounds good', r'listen up',
            r'loud and clear', r'rings a bell', r'word of mouth',
            r'tune in', r'strikes a chord', r'music to my ears'
        ]
        
        for phrase in auditory_phrases:
            if re.search(phrase, text):
                score += 0.5
        
        # Discussion and communication references
        communication_words = [
            r'discuss', r'conversation', r'dialogue', r'chat',
            r'interview', r'meeting', r'call', r'presentation'
        ]
        
        for word in communication_words:
            score += len(re.findall(word, text)) * 0.3
        
        return score
    
    def _extract_kinesthetic_patterns(self, text: str) -> float:
        """Extract kinesthetic learning patterns from text"""
        score = 0.0
        
        # Kinesthetic expressions
        kinesthetic_phrases = [
            r'hands-on', r'get a feel for', r'grasp the concept',
            r'concrete example', r'solid understanding', r'firm grasp',
            r'touch base', r'hands-on experience', r'learning by doing'
        ]
        
        for phrase in kinesthetic_phrases:
            if re.search(phrase, text):
                score += 0.5
        
        # Action and movement words
        action_words = [
            r'practice', r'exercise', r'experiment', r'build',
            r'create', r'make', r'construct', r'develop'
        ]
        
        for word in action_words:
            score += len(re.findall(word, text)) * 0.3
        
        return score
    
    def _get_context_weight(self, vak_type: str, text: str) -> float:
        """Apply contextual weighting based on sentence structure and context"""
        base_weight = 1.0
        
        # Boost score if the person is describing how they learn or understand
        learning_context_patterns = [
            r'i learn', r'i understand', r'i grasp', r'i get it',
            r'makes sense when', r'easier for me to', r'i prefer'
        ]
        
        has_learning_context = any(re.search(pattern, text) for pattern in learning_context_patterns)
        
        if has_learning_context:
            base_weight *= 1.3
        
        # Context-specific weighting
        if vak_type == 'visual':
            # Boost if talking about presentations, documents, or visual elements
            if re.search(r'presentation|document|slide|chart|visual', text):
                base_weight *= 1.2
                
        elif vak_type == 'auditory':
            # Boost if talking about discussions, explanations, or verbal communication
            if re.search(r'discussion|explain|talk about|verbal|meeting', text):
                base_weight *= 1.2
                
        elif vak_type == 'kinesthetic':
            # Boost if talking about practical application or hands-on activities
            if re.search(r'practical|hands-on|experience|try|practice', text):
                base_weight *= 1.2
        
        return base_weight
    
    def analyze_conversation_patterns(self, conversation_history: List[str]) -> Dict[str, float]:
        """
        Analyze VAK patterns across multiple conversation turns.
        
        Args:
            conversation_history: List of conversation turns
            
        Returns:
            Aggregated VAK scores
        """
        if not conversation_history:
            return {"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34}
        
        try:
            # Get predictions for each turn
            turn_predictions = []
            for turn in conversation_history:
                if turn.strip():  # Skip empty turns
                    pred = self.predict(turn)
                    turn_predictions.append(pred)
            
            if not turn_predictions:
                return {"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34}
            
            # Calculate weighted average (recent turns have more weight)
            total_visual = 0.0
            total_auditory = 0.0
            total_kinesthetic = 0.0
            total_weight = 0.0
            
            for i, pred in enumerate(turn_predictions):
                # Recent turns get higher weight
                weight = 1.0 + (i / len(turn_predictions)) * 0.5
                total_visual += pred["visual"] * weight
                total_auditory += pred["auditory"] * weight
                total_kinesthetic += pred["kinesthetic"] * weight
                total_weight += weight
            
            # Normalize
            return {
                "visual": total_visual / total_weight,
                "auditory": total_auditory / total_weight,
                "kinesthetic": total_kinesthetic / total_weight
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conversation patterns: {e}")
            return {"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34}
    
    def get_vak_insights(self, vak_scores: Dict[str, float]) -> Dict[str, str]:
        """
        Generate insights and recommendations based on VAK scores.
        
        Args:
            vak_scores: VAK preference scores
            
        Returns:
            Dictionary with insights and recommendations
        """
        try:
            # Find dominant learning style
            dominant_style = max(vak_scores.items(), key=lambda x: x[1])
            dominant_name, dominant_score = dominant_style
            
            # Generate insights
            insights = {
                "dominant_style": dominant_name,
                "confidence": dominant_score,
                "distribution": vak_scores
            }
            
            # Style-specific recommendations
            recommendations = {
                "visual": [
                    "Use diagrams, charts, and visual aids",
                    "Encourage mind mapping and visual note-taking",
                    "Provide written summaries and bullet points",
                    "Use color coding and visual organization"
                ],
                "auditory": [
                    "Engage in verbal discussions and explanations",
                    "Use storytelling and analogies",
                    "Encourage reading aloud and verbal processing",
                    "Provide audio resources and recordings"
                ],
                "kinesthetic": [
                    "Incorporate hands-on activities and practice",
                    "Use real-world examples and case studies",
                    "Encourage movement and physical engagement",
                    "Focus on practical application and experimentation"
                ]
            }
            
            insights["recommendations"] = recommendations.get(dominant_name, [])
            
            # Mixed style insights
            if dominant_score < 0.5:
                insights["mixed_style"] = True
                insights["secondary_styles"] = [
                    style for style, score in vak_scores.items() 
                    if style != dominant_name and score > 0.25
                ]
            else:
                insights["mixed_style"] = False
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating VAK insights: {e}")
            return {
                "dominant_style": "balanced",
                "confidence": 0.33,
                "distribution": vak_scores,
                "recommendations": ["Use a variety of teaching methods"]
            }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        return {
            "ml_model": "loaded" if self.ml_model else "not available",
            "vectorizer": "loaded" if self.vectorizer else "not available",
            "approach": "ML + rule-based" if self.ml_model else "rule-based only",
            "visual_keywords": len(self.VISUAL_KEYWORDS),
            "auditory_keywords": len(self.AUDITORY_KEYWORDS),
            "kinesthetic_keywords": len(self.KINESTHETIC_KEYWORDS),
            "model_path": str(self.model_path)
        }