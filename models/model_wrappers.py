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
        self.max_length = 128  # Will be adjusted based on model requirements
        self.keras_input_shape = None  # Store the expected input shape for Keras models
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
            
            # For Keras models, detect the expected input shape
            if hasattr(self.model, 'input_shape') and 'keras' in str(type(self.model)).lower():
                input_shape = self.model.input_shape
                if input_shape and len(input_shape) >= 2:
                    # input_shape is typically (None, sequence_length) for text models
                    expected_seq_length = input_shape[1]
                    if expected_seq_length and expected_seq_length != self.max_length:
                        logger.info(f"Adjusting max_length from {self.max_length} to {expected_seq_length} based on model input shape")
                        self.max_length = expected_seq_length
                    self.keras_input_shape = input_shape
                    logger.info(f"Keras model input shape: {input_shape}")
            
            self.is_loaded = True
            logger.info(f"Sarcasm model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load sarcasm model: {e}")
            self.is_loaded = False
    
    def preprocess_text_for_keras(self, text: str) -> np.ndarray:
        """Preprocess text specifically for Keras models"""
        if not text or not self.tokenizer:
            return np.array([])
        
        # Tokenize text and return only input_ids as numpy array
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None  # Return as lists, not tensors
        )
        
        # Return only input_ids as numpy array with proper shape for Keras
        input_ids = np.array([encoded['input_ids']])  # Shape: (1, max_length)
        return input_ids
    
    def preprocess_text_for_pytorch(self, text: str) -> Dict[str, Any]:
        """Preprocess text for PyTorch/Transformers models"""
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
            return_tensors='pt'  # Return as PyTorch tensors
        )
        
        return encoded
    
    def predict(self, text: Any) -> Tuple[bool, float]:
        try:
            # Convert input to string first, handling various input types
            if isinstance(text, list):
                # Join list elements into a single string
                text = " ".join(str(item).strip() for item in text)
            elif not isinstance(text, str):
                text = str(text)
            
            # Clean the text
            text = text.strip()
            
            logger.debug(f"[Sarcasm Input] Cleaned Text: {text}")

            if not self.is_loaded or not text:
                return False, 0.0

            # 1) Keras/TensorFlow models
            if hasattr(self.model, 'predict') and 'keras' in str(type(self.model)).lower():
                # Use specialized preprocessing for Keras
                input_data = self.preprocess_text_for_keras(text)
                if input_data.size == 0:
                    logger.error("Text preprocessing failed for Keras model")
                    return False, 0.0
                
                logger.debug(f"Keras input shape: {input_data.shape}, expected: {self.keras_input_shape}")
                
                # Verify input shape matches model expectations
                if self.keras_input_shape and len(self.keras_input_shape) >= 2:
                    expected_length = self.keras_input_shape[1]
                    if input_data.shape[1] != expected_length:
                        logger.error(f"Input shape mismatch: got {input_data.shape}, expected (*, {expected_length})")
                        return False, 0.0
                
                # Make prediction with Keras model
                predictions = self.model.predict(input_data, verbose=0)
                
                # Handle different output shapes
                if predictions.ndim > 1 and predictions.shape[1] > 1:
                    # Multi-class output: assume class 1 is "sarcastic"
                    sarcasm_prob = float(predictions[0][1])
                elif predictions.ndim > 1:
                    # Single output per sample
                    sarcasm_prob = float(predictions[0][0])
                else:
                    # Single value output
                    sarcasm_prob = float(predictions[0])

            # 2) sklearn models with vectorizer (TfidfVectorizer, CountVectorizer, etc.)
            elif hasattr(self.model, 'predict_proba') and hasattr(self, 'vectorizer') and self.vectorizer is not None:
                # Transform text using the vectorizer first
                text_vectorized = self.vectorizer.transform([text])
                probabilities = self.model.predict_proba(text_vectorized)
                
                # Handle different probability array shapes
                if probabilities.ndim > 1 and probabilities.shape[1] > 1:
                    # Binary classification: assume class 1 is "sarcastic"
                    sarcasm_prob = probabilities[0][1]
                else:
                    # Single probability value
                    sarcasm_prob = probabilities[0] if probabilities.ndim > 0 else probabilities

            # 3) sklearn models without explicit vectorizer (model handles text directly)
            elif hasattr(self.model, 'predict_proba'):
                # Some models might handle raw text directly
                try:
                    probabilities = self.model.predict_proba([text])
                    
                    if probabilities.ndim > 1 and probabilities.shape[1] > 1:
                        sarcasm_prob = probabilities[0][1]
                    else:
                        sarcasm_prob = probabilities[0] if probabilities.ndim > 0 else probabilities
                except Exception as e:
                    logger.error(f"Model predict_proba failed with raw text: {e}")
                    return False, 0.0

            # 4) Hugging‑Face/PyTorch model
            elif hasattr(self.model, 'forward'):
                processed = self.preprocess_text_for_pytorch(text)
                if not processed:
                    return False, 0.0
                    
                with torch.no_grad():
                    outputs = self.model(**processed)
                    probs = torch.softmax(outputs.logits, dim=1)[0]
                    sarcasm_prob = probs[1].item() if len(probs) > 1 else probs[0].item()

            # 5) Fallback custom model
            else:
                if hasattr(self, 'vectorizer') and self.vectorizer is not None:
                    text_vectorized = self.vectorizer.transform([text])
                    predictions = self.model.predict(text_vectorized)
                else:
                    predictions = self.model.predict([text])
                    
                sarcasm_prob = predictions[0] if hasattr(predictions, '__len__') else predictions

            # Ensure probability is in valid range
            sarcasm_prob = max(0.0, min(1.0, float(sarcasm_prob)))
            is_sarcastic = sarcasm_prob > 0.5
            
            logger.debug(f"Sarcasm prediction: {is_sarcastic}, confidence: {sarcasm_prob}")
            return is_sarcastic, sarcasm_prob

        except Exception as e:
            logger.error(f"Sarcasm prediction failed: {e}")
            logger.error(f"Input type: {type(text)}, Input value: {text}")
            logger.error(f"Model type: {type(self.model) if hasattr(self, 'model') else 'No model'}")
            logger.error(f"Has vectorizer: {hasattr(self, 'vectorizer') and self.vectorizer is not None}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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