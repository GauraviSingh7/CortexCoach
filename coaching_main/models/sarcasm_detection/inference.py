"""
Sarcasm Detection Model Interface
=================================

This module provides inference for detecting sarcasm in text conversation.
Combines ML models with rule-based pattern matching for robust detection.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import joblib
import torch
import re
from collections import Counter

logger = logging.getLogger(__name__)

class SarcasmDetectionModel:
    """
    Sarcasm detection model that analyzes text patterns, context, and linguistic features
    to identify sarcastic statements.
    """
    
    # Sarcasm indicator patterns
    SARCASM_PATTERNS = {
        'excessive_positivity': [
            r'\b(great|wonderful|fantastic|amazing|perfect|brilliant)\b.*\b(great|wonderful|fantastic|amazing|perfect|brilliant)\b',
            r'\b(oh\s+)?wow\b.*\b(amazing|great|perfect)\b',
            r'\bso\s+(excited|thrilled|happy)\b'
        ],
        'contradiction_markers': [
            r'\b(yeah\s+right|sure\s+thing|of\s+course|obviously)\b',
            r'\b(totally|absolutely|definitely)\b.*\b(not|never|no\s+way)\b',
            r'\bjust\s+what\s+i\s+(needed|wanted)\b'
        ],
        'ironic_phrases': [
            r'\bhow\s+(wonderful|great|nice|perfect)\b',
            r'\bthat\'?s\s+just\s+(great|perfect|wonderful)\b',
            r'\bi\s+love\s+it\s+when\b',
            r'\bexactly\s+what\s+i\s+(wanted|needed|hoped\s+for)\b'
        ],
        'exaggerated_responses': [
            r'\boh\s+(joy|goody|yay)\b',
            r'\bwell\s+isn\'?t\s+that\s+(special|nice|great)\b',
            r'\bhow\s+(delightful|charming|lovely)\b'
        ]
    }
    
    # Context-dependent sarcasm indicators
    NEGATIVE_CONTEXT_WORDS = [
        'problem', 'issue', 'wrong', 'broken', 'failed', 'disaster', 'mess',
        'terrible', 'awful', 'horrible', 'worst', 'hate', 'annoying', 'frustrating'
    ]
    
    POSITIVE_WORDS_IN_NEGATIVE_CONTEXT = [
        'great', 'wonderful', 'perfect', 'amazing', 'fantastic', 'excellent',
        'brilliant', 'awesome', 'lovely', 'nice', 'good'
    ]
    
    def __init__(self, model_path: Path):
        """
        Initialize the sarcasm detection model.
        
        Args:
            model_path: Path to the model directory
        """
        self.model_path = Path(model_path)
        self.ml_model = None
        self.vectorizer = None
        self.scaler = None
        self.context_model = None
        
        # Load model components
        self._load_models()
        
    def _load_models(self):
        """Load all model components"""
        try:
            # Load main sarcasm detection model
            ml_model_path = self.model_path / "sarcasm_classifier.pkl"
            if ml_model_path.exists():
                self.ml_model = joblib.load(ml_model_path)
                logger.info("Sarcasm ML classifier loaded successfully")
            
            # Load text vectorizer
            vectorizer_path = self.model_path / "vectorizer.pkl"
            if vectorizer_path.exists():
                self.vectorizer = joblib.load(vectorizer_path)
                logger.info("Sarcasm vectorizer loaded successfully")
            
            # Load feature scaler
            scaler_path = self.model_path / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Sarcasm scaler loaded successfully")
            
            # Load context-aware model (if available)
            context_model_path = self.model_path / "context_model.pth"
            if context_model_path.exists():
                self.context_model = torch.load(context_model_path, map_location='cpu')
                self.context_model.eval()
                logger.info("Sarcasm context model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading sarcasm models: {e}")
    
    def predict(self, text: str, context: Optional[List[str]] = None) -> float:
        """
        Predict sarcasm probability from text.
        
        Args:
            text: Input text to analyze
            context: Previous conversation context (optional)
            
        Returns:
            Sarcasm score between 0.0 (not sarcastic) and 1.0 (highly sarcastic)
        """
        try:
            # Clean and preprocess text
            text_clean = self._preprocess_text(text)
            
            if not text_clean.strip():
                return 0.0  # No sarcasm in empty text
            
            # Get ML prediction if model is available
            if self.ml_model and self.vectorizer:
                ml_score = self._predict_ml(text_clean)
            else:
                ml_score = 0.5  # Neutral if no ML model
            
            # Get rule-based prediction
            rule_score = self._predict_rule_based(text_clean, context)
            
            # Get context-aware prediction if available
            if self.context_model and context:
                context_score = self._predict_context_aware(text_clean, context)
                # Weighted combination: ML (40%), Rules (40%), Context (20%)
                combined_score = 0.4 * ml_score + 0.4 * rule_score + 0.2 * context_score
            else:
                # Weighted combination: ML (60%), Rules (40%)
                combined_score = 0.6 * ml_score + 0.4 * rule_score
            
            # Ensure score is in valid range
            return max(0.0, min(1.0, combined_score))
            
        except Exception as e:
            logger.error(f"Error in sarcasm prediction: {e}")
            return 0.0  # Conservative fallback
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Convert to lowercase and strip whitespace
        text_clean = text.lower().strip()
        
        # Normalize punctuation patterns that might indicate sarcasm
        text_clean = re.sub(r'\.{3,}', '...', text_clean)  # Multiple periods
        text_clean = re.sub(r'!{2,}', '!!', text_clean)    # Multiple exclamations
        text_clean = re.sub(r'\?{2,}', '??', text_clean)   # Multiple questions
        
        return text_clean
    
    def _predict_ml(self, text: str) -> float:
        """Predict using trained ML model"""
        try:
            # Vectorize text
            text_vector = self.vectorizer.transform([text])
            
            # Scale features if scaler is available
            if self.scaler:
                text_vector = self.scaler.transform(text_vector)
            
            # Get prediction
            if hasattr(self.ml_model, 'predict_proba'):
                # Binary classifier with probabilities
                probs = self.ml_model.predict_proba(text_vector)[0]
                # Assume classes are [not_sarcastic, sarcastic]
                sarcasm_prob = probs[1] if len(probs) > 1 else probs[0]
            else:
                # Regression model or binary prediction
                prediction = self.ml_model.predict(text_vector)[0]
                sarcasm_prob = float(prediction)
            
            return sarcasm_prob
            
        except Exception as e:
            logger.error(f"ML sarcasm prediction error: {e}")
            return 0.0
    
    def _predict_rule_based(self, text: str, context: Optional[List[str]] = None) -> float:
        """Predict using rule-based pattern matching"""
        try:
            sarcasm_score = 0.0
            
            # Check for sarcasm patterns
            pattern_score = self._analyze_sarcasm_patterns(text)
            sarcasm_score += pattern_score
            
            # Check for contradiction between sentiment and context
            contradiction_score = self._analyze_sentiment_contradiction(text, context)
            sarcasm_score += contradiction_score
            
            # Check for linguistic markers
            linguistic_score = self._analyze_linguistic_markers(text)
            sarcasm_score += linguistic_score
            
            # Check for timing and emphasis patterns
            emphasis_score = self._analyze_emphasis_patterns(text)
            sarcasm_score += emphasis_score
            
            # Normalize score
            return min(1.0, sarcasm_score)
            
        except Exception as e:
            logger.error(f"Rule-based sarcasm prediction error: {e}")
            return 0.0
    
    def _analyze_sarcasm_patterns(self, text: str) -> float:
        """Analyze text for known sarcasm patterns"""
        score = 0.0
        
        for category, patterns in self.SARCASM_PATTERNS.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                if matches > 0:
                    if category == 'excessive_positivity':
                        score += matches * 0.3
                    elif category == 'contradiction_markers':
                        score += matches * 0.4
                    elif category == 'ironic_phrases':
                        score += matches * 0.35
                    elif category == 'exaggerated_responses':
                        score += matches * 0.25
        
        return min(0.6, score)  # Cap at 0.6 for pattern matching
    
    def _analyze_sentiment_contradiction(self, text: str, context: Optional[List[str]] = None) -> float:
        """Analyze contradiction between expressed sentiment and context"""
        score = 0.0
        
        # Check for positive words in negative context within the same text
        has_negative_context = any(word in text for word in self.NEGATIVE_CONTEXT_WORDS)
        has_positive_words = any(word in text for word in self.POSITIVE_WORDS_IN_NEGATIVE_CONTEXT)
        
        if has_negative_context and has_positive_words:
            score += 0.4  # Strong indicator of sarcasm
        
        # Check context from previous messages if available
        if context:
            recent_context = ' '.join(context[-3:]).lower()  # Last 3 messages
            context_is_negative = any(word in recent_context for word in self.NEGATIVE_CONTEXT_WORDS)
            current_is_positive = any(word in text for word in self.POSITIVE_WORDS_IN_NEGATIVE_CONTEXT)
            
            if context_is_negative and current_is_positive:
                score += 0.3  # Contextual sarcasm indicator
        
        return score
    
    def _analyze_linguistic_markers(self, text: str) -> float:
        """Analyze linguistic markers that often indicate sarcasm"""
        score = 0.0
        
        # Quotation marks around positive words (e.g., "great" job)
        quoted_positive = re.findall(r'["\'](\w+)["\']', text)
        if any(word in self.POSITIVE_WORDS_IN_NEGATIVE_CONTEXT for word in quoted_positive):
            score += 0.2
        
        # Exaggerated punctuation
        if re.search(r'\.{3,}', text):  # Multiple periods
            score += 0.1
        if re.search(r'!{2,}', text):   # Multiple exclamations
            score += 0.15
        
        # ALL CAPS words (could indicate sarcastic emphasis)
        caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
        if caps_words:
            score += min(0.2, len(caps_words) * 0.05)
        
        # Repetitive letters (e.g., "sooo great")
        if re.search(r'(\w)\1{2,}', text):
            score += 0.1
        
        # Question tags that might indicate sarcasm ("Nice job, right?")
        if re.search(r',\s*(right|huh|eh)\?', text):
            score += 0.15
        
        return score
    
    def _analyze_emphasis_patterns(self, text: str) -> float:
        """Analyze emphasis patterns that might indicate sarcasm"""
        score = 0.0
        
        # Overuse of intensifiers
        intensifiers = ['very', 'really', 'so', 'such', 'quite', 'absolutely', 'totally']
        intensifier_count = sum(len(re.findall(rf'\b{intensifier}\b', text, re.IGNORECASE)) 
                               for intensifier in intensifiers)
        
        if intensifier_count >= 2:
            score += 0.2  # Multiple intensifiers might indicate sarcasm
        
        # Delayed reactions or responses
        if re.search(r'\b(oh\s+)?wow\b', text, re.IGNORECASE):
            score += 0.1
        
        # Fake enthusiasm markers
        fake_enthusiasm = ['yay', 'woohoo', 'hooray', 'goody']
        if any(word in text.lower() for word in fake_enthusiasm):
            score += 0.15
        
        return score
    
    def _predict_context_aware(self, text: str, context: List[str]) -> float:
        """Predict using context-aware model (if available)"""
        try:
            if not self.context_model:
                return 0.5
            
            # Prepare context + current text for model input
            context_text = ' [SEP] '.join(context[-5:] + [text])  # Last 5 context messages
            
            # This is a placeholder - implement based on your context model architecture
            # You might need to tokenize, encode, and process the text appropriately
            
            # For now, return a simplified context-based score
            return self._simplified_context_analysis(text, context)
            
        except Exception as e:
            logger.error(f"Context-aware prediction error: {e}")
            return 0.5
    
    def _simplified_context_analysis(self, text: str, context: List[str]) -> float:
        """Simplified context analysis as fallback"""
        score = 0.0
        
        if not context:
            return 0.5
        
        # Check if response seems inappropriately positive given context
        recent_context = ' '.join(context[-3:]).lower()
        
        # Count negative sentiment in context
        negative_count = sum(1 for word in self.NEGATIVE_CONTEXT_WORDS if word in recent_context)
        
        # Count positive words in current response
        positive_count = sum(1 for word in self.POSITIVE_WORDS_IN_NEGATIVE_CONTEXT if word in text.lower())
        
        # If context is negative but response is overly positive, might be sarcastic
        if negative_count > 0 and positive_count > 0:
            score = min(0.7, (negative_count * positive_count) * 0.2)
        
        return score
    
    def analyze_conversation_sarcasm(self, conversation_history: List[Dict]) -> Dict[str, any]:
        """
        Analyze sarcasm patterns across entire conversation.
        
        Args:
            conversation_history: List of conversation turns with text and metadata
            
        Returns:
            Dictionary with sarcasm analysis results
        """
        if not conversation_history:
            return {"overall_sarcasm": 0.0, "sarcastic_turns": []}
        
        try:
            sarcasm_scores = []
            sarcastic_turns = []
            
            for i, turn in enumerate(conversation_history):
                text = turn.get('text', '')
                if not text.strip():
                    continue
                
                # Get context from previous turns
                context = [t.get('text', '') for t in conversation_history[:i] if t.get('text', '').strip()]
                
                # Predict sarcasm for this turn
                sarcasm_score = self.predict(text, context)
                sarcasm_scores.append(sarcasm_score)
                
                # Mark as sarcastic if score is above threshold
                if sarcasm_score > 0.6:
                    sarcastic_turns.append({
                        'turn_index': i,
                        'text': text,
                        'sarcasm_score': sarcasm_score,
                        'speaker': turn.get('speaker', 'unknown')
                    })
            
            # Calculate overall statistics
            overall_sarcasm = np.mean(sarcasm_scores) if sarcasm_scores else 0.0
            max_sarcasm = max(sarcasm_scores) if sarcasm_scores else 0.0
            sarcasm_frequency = len(sarcastic_turns) / len(conversation_history) if conversation_history else 0.0
            
            return {
                "overall_sarcasm": float(overall_sarcasm),
                "max_sarcasm": float(max_sarcasm),
                "sarcasm_frequency": float(sarcasm_frequency),
                "total_turns": len(conversation_history),
                "sarcastic_turns": sarcastic_turns,
                "scores": sarcasm_scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conversation sarcasm: {e}")
            return {"overall_sarcasm": 0.0, "sarcastic_turns": []}
    
    def get_sarcasm_explanation(self, text: str, score: float) -> Dict[str, any]:
        """
        Provide explanation for sarcasm detection result.
        
        Args:
            text: Input text
            score: Sarcasm score
            
        Returns:
            Dictionary with explanation of the detection
        """
        try:
            explanation = {
                "sarcasm_score": score,
                "likely_sarcastic": score > 0.5,
                "confidence": "high" if abs(score - 0.5) > 0.3 else "medium" if abs(score - 0.5) > 0.1 else "low",
                "indicators": []
            }
            
            # Check which patterns contributed to the score
            text_lower = text.lower()
            
            # Pattern indicators
            for category, patterns in self.SARCASM_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        explanation["indicators"].append(f"Sarcasm pattern detected: {category}")
                        break
            
            # Sentiment contradiction
            has_negative = any(word in text_lower for word in self.NEGATIVE_CONTEXT_WORDS)
            has_positive = any(word in text_lower for word in self.POSITIVE_WORDS_IN_NEGATIVE_CONTEXT)
            if has_negative and has_positive:
                explanation["indicators"].append("Sentiment contradiction detected")
            
            # Linguistic markers
            if re.search(r'["\'](\w+)["\']', text):
                explanation["indicators"].append("Quoted words detected")
            if re.search(r'\.{3,}|!{2,}', text):
                explanation["indicators"].append("Exaggerated punctuation")
            if re.search(r'\b[A-Z]{2,}\b', text):
                explanation["indicators"].append("Emphasis through capitalization")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating sarcasm explanation: {e}")
            return {
                "sarcasm_score": score,
                "likely_sarcastic": False,
                "confidence": "low",
                "indicators": ["Error in analysis"]
            }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        return {
            "ml_model": "loaded" if self.ml_model else "not available",
            "vectorizer": "loaded" if self.vectorizer else "not available",
            "context_model": "loaded" if self.context_model else "not available",
            "approach": "ML + rule-based + context" if self.context_model else "ML + rule-based" if self.ml_model else "rule-based only",
            "pattern_categories": len(self.SARCASM_PATTERNS),
            "negative_context_words": len(self.NEGATIVE_CONTEXT_WORDS),
            "positive_indicator_words": len(self.POSITIVE_WORDS_IN_NEGATIVE_CONTEXT),
            "model_path": str(self.model_path)
        }