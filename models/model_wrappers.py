import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from typing import Dict, Any, Optional, List, Tuple
import logging
from abc import ABC, abstractmethod
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

logger = logging.getLogger(__name__)

class ModelWrapper(ABC):
    """Base class for all model wrappers"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.last_prediction = None
    
    @abstractmethod
    def load_model(self):
        """Load the model - implemented by subclasses"""
        pass
    
    @abstractmethod
    def predict(self, input_data):
        """Make prediction - implemented by subclasses"""
        pass
    
    def is_model_loaded(self) -> bool:
        return self.is_loaded

class FacialEmotionWrapper(ModelWrapper):
    """Wrapper for facial emotion recognition model (Keras .h5)"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        # Adjust these based on your model's training
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.input_shape = (48, 48)  # Common for emotion models
        self.load_model()
    
    def load_model(self):
        """Load the Keras emotion model"""
        try:
            self.model = load_model(self.model_path)
            self.is_loaded = True
            logger.info(f"Facial emotion model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load facial emotion model: {e}")
            self.is_loaded = False
    
    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess image for emotion detection"""
        if image is None or not isinstance(image, np.ndarray):
            logger.warning("Invalid image input for preprocessing")
            return None

        try:
            # Fix object dtype (e.g., from uploaded PIL images)
            if image.dtype == object:
                image = np.array(image.tolist(), dtype=np.uint8)

            # Convert to grayscale if RGB/BGR
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize to model input size
            logger.debug(f"Image dtype: {image.dtype}, shape: {image.shape}")

            image = cv2.resize(image, self.input_shape)

            # Normalize and reshape
            image = image.astype("float32") / 255.0
            image = np.expand_dims(image, axis=(0, -1))

            return image

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None


    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """Predict emotion from facial image"""
        if not self.is_loaded:
            logger.warning("Facial emotion model not loaded")
            return {label: 0.0 for label in self.emotion_labels}
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return {label: 0.0 for label in self.emotion_labels}
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Convert to emotion dictionary
            emotion_scores = {}
            for i, label in enumerate(self.emotion_labels):
                emotion_scores[label] = float(predictions[0][i])
            
            self.last_prediction = emotion_scores
            return emotion_scores
            
        except Exception as e:
            logger.error(f"Facial emotion prediction failed: {e}")
            return {label: 0.0 for label in self.emotion_labels}

class SarcasmDetectionWrapper(ModelWrapper):
    """Wrapper for BERT-based sarcasm detection model (.pkl)"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.tokenizer = None
        self.max_length = 128  # Adjust based on your model
        self.load_model()
    
    def load_model(self):
        """Load the pickled sarcasm model"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle different pickle formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.tokenizer = model_data.get('tokenizer')
                self.max_length = model_data.get('max_length', 128)
            else:
                self.model = model_data
                # Initialize default tokenizer if not pickled
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            self.is_loaded = True
            logger.info(f"Sarcasm model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load sarcasm model: {e}")
            self.is_loaded = False
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Preprocess text for sarcasm detection"""
        if not text or not self.tokenizer:
            return {}
        
        # Tokenize text
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt' if hasattr(self.model, 'forward') else None
        )
        
        return encoded
    
    def predict(self, text: str) -> Tuple[bool, float]:
        """Predict sarcasm in text"""
        # 🚨 Defensive fix: always coerce list inputs into one string
        if isinstance(text, list):
            text = " ".join(text)

        # Now ensure model is loaded and text is non-empty
        if not self.is_loaded or not text:
            return False, 0.0

        try:
            # Preprocess text
            processed = self.preprocess_text(text)
            if not processed:
                return False, 0.0

            # Make prediction based on model type
            if hasattr(self.model, 'predict_proba'):
                # Sklearn-like model
                if 'input_ids' in processed:
                    input_data = processed['input_ids'].numpy().flatten()[:self.max_length]
                else:
                    input_data = text  # Use raw text directly
                probabilities = self.model.predict_proba([input_data])
                sarcasm_prob = probabilities[0][1] if len(probabilities[0]) > 1 else probabilities[0][0]

            elif hasattr(self.model, 'forward'):
                # PyTorch BERT model
                with torch.no_grad():
                    outputs = self.model(**processed)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    sarcasm_prob = probabilities[0][1].item()

            else:
                # Custom model - attempt direct prediction
                sarcasm_prob = self.model.predict([text])[0]

            is_sarcastic = sarcasm_prob > 0.5
            confidence = float(sarcasm_prob)
            self.last_prediction = (is_sarcastic, confidence)
            return is_sarcastic, confidence

        except Exception as e:
            logger.error(f"Sarcasm prediction failed: {e}")
            return False, 0.0
        
        
class VAKLearningStyleWrapper:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
            self.model.eval()

            # Assuming labels = ['visual', 'auditory', 'reading', 'kinesthetic']
            self.label_map = {0: "visual", 1: "auditory", 2: "reading", 3: "kinesthetic"}

            logger.info(f"VAK model loaded from {self.model_dir}")
            self.loaded = True
        except Exception as e:
            logger.error(f"Failed to load VAK model: {e}")
            self.loaded = False

    def predict(self, text: str):
        if not self.loaded:
            return "visual", 0.25

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                predicted_label = np.argmax(probs)
                return self.label_map[predicted_label], float(probs[predicted_label])
        except Exception as e:
            logger.error(f"VAK prediction failed: {e}")
            return "visual", 0.25