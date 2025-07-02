from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging
from .model_wrappers import FacialEmotionWrapper, SarcasmDetectionWrapper, VAKLearningStyleWrapper

logger = logging.getLogger(__name__)

class MultimodalInferencePipeline:
    """Unified pipeline for running all model inferences"""
    
    def __init__(self, model_paths: Dict[str, str]):
        """
        Initialize with paths to all models
        
        Args:
            model_paths: Dict with keys 'facial_emotion', 'sarcasm', 'vak'
        """
        self.models = {}
        self.load_all_models(model_paths)
    
    def load_all_models(self, model_paths: Dict[str, str]):
        """Load all models with error handling"""
        
        # Load facial emotion model
        try:
            self.models['facial_emotion'] = FacialEmotionWrapper(
                model_paths['facial_emotion']
            )
        except Exception as e:
            logger.error(f"Failed to load facial emotion model: {e}")
            self.models['facial_emotion'] = None
        
        # Load sarcasm detection model
        try:
            self.models['sarcasm'] = SarcasmDetectionWrapper(
                model_paths['sarcasm']
            )
        except Exception as e:
            logger.error(f"Failed to load sarcasm model: {e}")
            self.models['sarcasm'] = None
        
        # Load VAK learning style model
        try:
            self.models['vak'] = VAKLearningStyleWrapper(
                model_paths['vak']
            )
        except Exception as e:
            logger.error(f"Failed to load VAK model: {e}")
            self.models['vak'] = None
    
    def process_multimodal_input(
        self, 
        text: Optional[str] = None,
        image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Process all available inputs through respective models
        
        Args:
            text: User utterance text (string or list of strings)
            image: Webcam image (numpy array)
        
        Returns:
            Dict containing all model outputs
        """
        # Normalize text input consistently
        if isinstance(text, list):
            text = ' '.join(str(item).strip() for item in text if item)
        elif text is not None:
            text = str(text).strip()
        
        # Skip empty text
        if not text:
            text = None
        
        logger.debug(f"[Input Text Type] {type(text)} | Value: {text[:50] if text else 'None'}")

        results = {
            'facial_emotion': {},
            'sarcasm_detected': False,
            'sarcasm_confidence': 0.0,
            'vak_type': 'visual',
            'vak_confidence': 0.33,
            'processing_success': {
                'facial_emotion': False,
                'sarcasm': False,
                'vak': False
            }
        }
        
        # Process facial emotion from image
        if image is not None and self.models.get('facial_emotion'):
            try:
                emotion_scores = self.models['facial_emotion'].predict(image)
                results['facial_emotion'] = emotion_scores
                results['processing_success']['facial_emotion'] = True
                logger.debug(f"Facial emotion processed: {emotion_scores}")
            except Exception as e:
                logger.error(f"Facial emotion processing failed: {e}")
        
        # Process text for sarcasm detection
        if text and self.models.get('sarcasm'):
            try:
                is_sarcastic, confidence = self.models['sarcasm'].predict(text)
                results['sarcasm_detected'] = is_sarcastic
                results['sarcasm_confidence'] = confidence
                results['processing_success']['sarcasm'] = True
                logger.debug(f"Sarcasm processed: {is_sarcastic}, {confidence}")
            except Exception as e:
                logger.error(f"Sarcasm processing failed: {e}")
        
        # Process text for VAK learning style
        if text and self.models.get('vak'):
            try:
                vak_style, confidence = self.models['vak'].predict(text)
                results['vak_type'] = vak_style
                results['vak_confidence'] = confidence
                results['processing_success']['vak'] = True
                logger.debug(f"VAK processed: {vak_style}, {confidence}")
            except Exception as e:
                logger.error(f"VAK processing failed: {e}")
        
        return results
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get loading status of all models"""
        return {
            'facial_emotion': self.models.get('facial_emotion') is not None and 
                            self.models['facial_emotion'].is_model_loaded(),
            'sarcasm': self.models.get('sarcasm') is not None and
                      self.models['sarcasm'].is_model_loaded(),
            'vak': self.models.get('vak') is not None and
                  self.models['vak'].is_model_loaded()
        }
    
    def reload_model(self, model_name: str, model_path: str) -> bool:
        """Reload a specific model"""
        try:
            if model_name == 'facial_emotion':
                self.models['facial_emotion'] = FacialEmotionWrapper(model_path)
            elif model_name == 'sarcasm':
                self.models['sarcasm'] = SarcasmDetectionWrapper(model_path)
            elif model_name == 'vak':
                self.models['vak'] = VAKLearningStyleWrapper(model_path)
            else:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to reload {model_name} model: {e}")
            return False
        
    def process_facial_emotion(self, image: np.ndarray) -> Dict[str, float]:
        """
        Process facial emotion independently from image.
        Returns a dict of emotion scores.
        """
        if self.models.get('facial_emotion') is None:
            raise RuntimeError("Facial emotion model is not loaded.")
        
        try:
            return self.models['facial_emotion'].predict(image)
        except Exception as e:
            logger.error(f"Facial emotion processing error: {e}")
            return {}