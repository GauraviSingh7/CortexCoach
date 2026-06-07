"""
Speech Emotion Recognition Model Interface
==========================================

This module provides a clean interface for the emotion recognition model.
Adapt this template to match your specific model architecture and requirements.
"""

import numpy as np
import logging
from typing import Dict, Optional, Union
from pathlib import Path
import joblib
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class EmotionRecognitionModel:
    """
    Speech Emotion Recognition model wrapper.
    Handles both audio and text-based emotion recognition.
    """
    
    # Define emotion labels - adjust based on your model's output
    EMOTION_LABELS = [
        "happy", "sad", "angry", "fear", "surprise", 
        "disgust", "neutral", "calm", "excited", "frustrated"
    ]
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize the emotion recognition model.
        
        Args:
            model_path: Path to the model directory or file
        """
        self.model_path = Path(model_path)
        self.audio_model = None
        self.text_model = None
        self.tokenizer = None
        self.scaler = None
        self.label_encoder = None
        
        # Load all model components
        self._load_models()
        
    def _load_models(self):
        """Load all model components with error handling"""
        try:
            # Load audio-based emotion model (assuming it's a PyTorch model)
            audio_model_path = self.model_path / "audio_emotion_model.pth"
            if audio_model_path.exists():
                self.audio_model = torch.load(audio_model_path, map_location='cpu')
                self.audio_model.eval()
                logger.info("Audio emotion model loaded successfully")
            
            # Load text-based emotion model (assuming it's a scikit-learn model)
            text_model_path = self.model_path / "text_emotion_model.pkl"
            if text_model_path.exists():
                self.text_model = joblib.load(text_model_path)
                logger.info("Text emotion model loaded successfully")
            
            # Load tokenizer if using transformer-based approach
            tokenizer_path = self.model_path / "tokenizer"
            if tokenizer_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                logger.info("Tokenizer loaded successfully")
            
            # Load feature scaler
            scaler_path = self.model_path / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Feature scaler loaded successfully")
            
            # Load label encoder
            label_encoder_path = self.model_path / "label_encoder.pkl"
            if label_encoder_path.exists():
                self.label_encoder = joblib.load(label_encoder_path)
                # Update emotion labels from the encoder
                if hasattr(self.label_encoder, 'classes_'):
                    self.EMOTION_LABELS = self.label_encoder.classes_.tolist()
                logger.info("Label encoder loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading emotion models: {e}")
            raise
    
    def predict(self, text: str, audio_data: Optional[bytes] = None) -> Dict[str, float]:
        """
        Predict emotions from text and/or audio data.
        
        Args:
            text: Transcript text
            audio_data: Raw audio bytes (optional)
            
        Returns:
            Dictionary mapping emotion names to confidence scores
        """
        try:
            # Initialize emotion probabilities
            emotion_probs = {emotion: 0.0 for emotion in self.EMOTION_LABELS}
            
            # Text-based emotion prediction
            if text and (self.text_model is not None or self.tokenizer is not None):
                text_probs = self._predict_from_text(text)
                for emotion, prob in text_probs.items():
                    emotion_probs[emotion] += prob * 0.7  # Weight text predictions
            
            # Audio-based emotion prediction
            if audio_data and self.audio_model is not None:
                audio_probs = self._predict_from_audio(audio_data)
                for emotion, prob in audio_probs.items():
                    emotion_probs[emotion] += prob * 0.3  # Weight audio predictions
            
            # If only text is available, give it full weight
            if text and not audio_data:
                text_probs = self._predict_from_text(text)
                emotion_probs = text_probs
            
            # Normalize probabilities to sum to 1
            total_prob = sum(emotion_probs.values())
            if total_prob > 0:
                emotion_probs = {k: v/total_prob for k, v in emotion_probs.items()}
            else:
                # Fallback to neutral if no predictions
                emotion_probs = {"neutral": 1.0}
            
            return emotion_probs
            
        except Exception as e:
            logger.error(f"Error in emotion prediction: {e}")
            return {"neutral": 0.6, "uncertain": 0.4}
    
    def _predict_from_text(self, text: str) -> Dict[str, float]:
        """Predict emotions from text using transformer or traditional ML model"""
        try:
            if self.tokenizer and hasattr(self.text_model, 'predict_proba'):
                # Transformer-based approach
                return self._predict_transformer_text(text)
            elif self.text_model:
                # Traditional ML approach (e.g., TF-IDF + classifier)
                return self._predict_traditional_text(text)
            else:
                logger.warning("No text model available")
                return {"neutral": 1.0}
                
        except Exception as e:
            logger.error(f"Text prediction error: {e}")
            return {"neutral": 1.0}
    
    def _predict_transformer_text(self, text: str) -> Dict[str, float]:
        """Predict using transformer-based text model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)[0]
                
            # Map to emotion labels
            emotion_probs = {}
            for i, emotion in enumerate(self.EMOTION_LABELS):
                if i < len(probs):
                    emotion_probs[emotion] = float(probs[i])
                else:
                    emotion_probs[emotion] = 0.0
                    
            return emotion_probs
            
        except Exception as e:
            logger.error(f"Transformer text prediction error: {e}")
            return {"neutral": 1.0}
    
    def _predict_traditional_text(self, text: str) -> Dict[str, float]:
        """Predict using traditional ML text model (e.g., sklearn)"""
        try:
            # If you have a vectorizer, use it
            vectorizer_path = self.model_path / "vectorizer.pkl"
            if vectorizer_path.exists():
                vectorizer = joblib.load(vectorizer_path)
                text_features = vectorizer.transform([text])
            else:
                # Simple fallback: basic text features
                text_features = self._extract_basic_text_features(text)
                if self.scaler:
                    text_features = self.scaler.transform([text_features])
            
            # Get predictions
            if hasattr(self.text_model, 'predict_proba'):
                probs = self.text_model.predict_proba(text_features)[0]
            else:
                # Binary classifier - convert to probabilities
                prediction = self.text_model.predict(text_features)[0]
                probs = np.zeros(len(self.EMOTION_LABELS))
                if prediction < len(probs):
                    probs[prediction] = 1.0
            
            # Map to emotion labels
            emotion_probs = {}
            for i, emotion in enumerate(self.EMOTION_LABELS):
                if i < len(probs):
                    emotion_probs[emotion] = float(probs[i])
                else:
                    emotion_probs[emotion] = 0.0
                    
            return emotion_probs
            
        except Exception as e:
            logger.error(f"Traditional text prediction error: {e}")
            return {"neutral": 1.0}
    
    def _extract_basic_text_features(self, text: str) -> np.ndarray:
        """Extract basic text features as fallback"""
        # Simple features - you can enhance this
        features = [
            len(text),  # Text length
            len(text.split()),  # Word count
            text.count('!'),  # Exclamation marks
            text.count('?'),  # Question marks
            text.count('.'),  # Periods
            len([w for w in text.split() if w.isupper()]),  # Uppercase words
        ]
        return np.array(features)
    
    def _predict_from_audio(self, audio_data: bytes) -> Dict[str, float]:
        """Predict emotions from audio data"""
        try:
            # Convert audio bytes to features
            audio_features = self._extract_audio_features(audio_data)
            
            if audio_features is None:
                logger.warning("Could not extract audio features")
                return {"neutral": 1.0}
            
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
                    outputs = self.audio_model(features_tensor)
                    probs = torch.softmax(outputs, dim=-1)[0]
                    probs = probs.numpy()
            else:
                # Sklearn model
                if hasattr(self.audio_model, 'predict_proba'):
                    probs = self.audio_model.predict_proba(audio_features)[0]
                else:
                    prediction = self.audio_model.predict(audio_features)[0]
                    probs = np.zeros(len(self.EMOTION_LABELS))
                    if prediction < len(probs):
                        probs[prediction] = 1.0
            
            # Map to emotion labels
            emotion_probs = {}
            for i, emotion in enumerate(self.EMOTION_LABELS):
                if i < len(probs):
                    emotion_probs[emotion] = float(probs[i])
                else:
                    emotion_probs[emotion] = 0.0
                    
            return emotion_probs
            
        except Exception as e:
            logger.error(f"Audio prediction error: {e}")
            return {"neutral": 1.0}
    
    def _extract_audio_features(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Extract audio features from raw audio data.
        This is a placeholder - implement based on your audio processing pipeline.
        """
        try:
            # Placeholder implementation
            # You would typically:
            # 1. Convert bytes to audio signal
            # 2. Extract features like MFCC, spectrograms, etc.
            # 3. Return feature vector
            
            # For now, return dummy features
            logger.warning("Using dummy audio features - implement actual feature extraction")
            return np.random.rand(40)  # Assuming 40-dimensional features
            
        except Exception as e:
            logger.error(f"Audio feature extraction error: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about loaded models"""
        return {
            "audio_model": "loaded" if self.audio_model else "not available",
            "text_model": "loaded" if self.text_model else "not available", 
            "tokenizer": "loaded" if self.tokenizer else "not available",
            "emotion_labels": str(self.EMOTION_LABELS),
            "model_path": str(self.model_path)
        }