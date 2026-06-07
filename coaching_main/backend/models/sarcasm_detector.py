"""
Sarcasm Detection Module for AI Coaching Observer
File: backend/models/sarcasm_detector.py
Complete implementation with rule-based detection
"""

import re
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class SarcasmDetector:
    """
    Detects sarcasm in coaching conversations using linguistic patterns,
    sentiment incongruence, and contextual analysis
    """
    
    def __init__(self):
        # Sarcasm indicators - phrases often used sarcastically
        self.sarcasm_phrases = [
            "oh great", "oh wonderful", "oh perfect", "oh fantastic",
            "yeah right", "sure sure", "of course", "obviously",
            "how surprising", "what a shock", "who would have thought",
            "thanks for nothing", "that's just great", "well done",
            "brilliant", "genius", "nice job", "smooth move",
            "real smart", "big help", "lot of good", "fat chance"
        ]
        
        # Exaggeration patterns often used sarcastically
        self.exaggeration_patterns = [
            r"\b(so|very|really|extremely|incredibly|absolutely|totally)\s+(helpful|useful|great|perfect|amazing)\b",
            r"\bjust\s+(perfect|great|wonderful|fantastic|brilliant)\b",
            r"\b(best|worst)\s+\w+\s+(ever|in the world)\b"
        ]
        
        # Negative sentiment words (for incongruence detection)
        self.negative_words = [
            "awful", "terrible", "horrible", "worst", "useless", "pointless",
            "stupid", "ridiculous", "waste", "joke", "disaster", "mess",
            "fail", "failed", "failing", "broken", "problem", "issue"
        ]
        
        # Positive sentiment words (for incongruence detection)
        self.positive_words = [
            "great", "wonderful", "fantastic", "perfect", "amazing", "excellent",
            "brilliant", "awesome", "superb", "outstanding", "terrific", "fabulous"
        ]
        
        # Context indicators - situations where sarcasm is more likely
        self.context_indicators = {
            'frustration': ["frustrated", "annoying", "annoyed", "irritated", "bothered"],
            'disappointment': ["disappointed", "expected", "hoped", "thought"],
            'criticism': ["should", "could have", "supposed to", "meant to"],
            'resistance': ["but", "however", "though", "still", "yet"]
        }
    
    def detect_sarcasm(self, text: str, conversation_history: List[str] = None) -> Tuple[float, str]:
        """
        Detect sarcasm in text
        
        Args:
            text: Text to analyze
            conversation_history: Previous messages for context (last 3-5 messages)
        
        Returns:
            (score, explanation) where score is 0.0-1.0 and explanation describes detection
        """
        text_lower = text.lower()
        score = 0.0
        indicators = []
        
        # 1. Check for explicit sarcasm phrases
        phrase_score = self._check_sarcasm_phrases(text_lower)
        if phrase_score > 0:
            score += phrase_score
            indicators.append("sarcastic phrase detected")
        
        # 2. Check for exaggeration patterns
        exaggeration_score = self._check_exaggeration(text_lower)
        if exaggeration_score > 0:
            score += exaggeration_score
            indicators.append("exaggeration pattern")
        
        # 3. Check for sentiment incongruence
        incongruence_score = self._check_sentiment_incongruence(text_lower)
        if incongruence_score > 0:
            score += incongruence_score
            indicators.append("sentiment incongruence")
        
        # 4. Check for contextual sarcasm indicators
        context_score = self._check_context_indicators(text_lower)
        if context_score > 0:
            score += context_score
            indicators.append("contextual indicators")
        
        # 5. Check for punctuation patterns (!!!, ???, excessive emphasis)
        punctuation_score = self._check_punctuation_patterns(text)
        if punctuation_score > 0:
            score += punctuation_score
            indicators.append("emphasis patterns")
        
        # 6. Analyze conversation history for mood/tone shift
        if conversation_history:
            history_score = self._analyze_conversation_history(text_lower, conversation_history)
            if history_score > 0:
                score += history_score
                indicators.append("tone shift detected")
        
        # Normalize score to 0.0-1.0
        score = min(score, 1.0)
        
        # Generate explanation
        if score > 0.7:
            confidence = "high"
        elif score > 0.4:
            confidence = "moderate"
        elif score > 0.2:
            confidence = "low"
        else:
            confidence = "minimal"
        
        explanation = f"{confidence} sarcasm likelihood"
        if indicators:
            explanation += f" ({', '.join(indicators)})"
        
        return score, explanation
    
    def _check_sarcasm_phrases(self, text: str) -> float:
        """Check for common sarcastic phrases"""
        score = 0.0
        
        for phrase in self.sarcasm_phrases:
            if phrase in text:
                score += 0.3
                logger.debug(f"Sarcasm phrase found: {phrase}")
        
        return min(score, 0.6)  # Cap at 0.6
    
    def _check_exaggeration(self, text: str) -> float:
        """Check for exaggeration patterns"""
        score = 0.0
        
        for pattern in self.exaggeration_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.25
                logger.debug(f"Exaggeration pattern matched")
        
        return min(score, 0.5)
    
    def _check_sentiment_incongruence(self, text: str) -> float:
        """
        Check if positive words appear with negative context
        This is a key indicator of sarcasm
        """
        score = 0.0
        
        has_positive = any(word in text for word in self.positive_words)
        has_negative = any(word in text for word in self.negative_words)
        
        # If both positive and negative sentiment in same utterance = likely sarcasm
        if has_positive and has_negative:
            score += 0.4
            logger.debug("Sentiment incongruence detected")
        
        # Check for "positive" words with negative intensifiers
        negative_intensifiers = ["not", "hardly", "barely", "scarcely", "never"]
        for intensifier in negative_intensifiers:
            for pos_word in self.positive_words:
                pattern = f"{intensifier}\\s+\\w+\\s+{pos_word}|{intensifier}\\s+{pos_word}"
                if re.search(pattern, text):
                    score += 0.3
                    logger.debug(f"Negative intensifier + positive word")
        
        return min(score, 0.6)
    
    def _check_context_indicators(self, text: str) -> float:
        """Check for contextual indicators that suggest sarcasm"""
        score = 0.0
        
        for context_type, keywords in self.context_indicators.items():
            if any(keyword in text for keyword in keywords):
                # If frustration/disappointment words appear with positive words
                has_positive = any(word in text for word in self.positive_words)
                if has_positive:
                    score += 0.3
                    logger.debug(f"Context indicator: {context_type} with positive words")
                else:
                    score += 0.1
        
        return min(score, 0.4)
    
    def _check_punctuation_patterns(self, text: str) -> float:
        """Check for excessive punctuation often used in sarcasm"""
        score = 0.0
        
        # Multiple exclamation marks
        if re.search(r'!{2,}', text):
            score += 0.15
            logger.debug("Multiple exclamation marks detected")
        
        # Multiple question marks
        if re.search(r'\?{2,}', text):
            score += 0.15
            logger.debug("Multiple question marks detected")
        
        # Ellipsis (often used sarcastically)
        if '...' in text or re.search(r'\.{3,}', text):
            score += 0.1
            logger.debug("Ellipsis detected")
        
        # ALL CAPS words (can indicate sarcasm/emphasis)
        caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
        if caps_words:
            score += 0.1 * min(len(caps_words), 3)
            logger.debug(f"ALL CAPS detected")
        
        return min(score, 0.4)
    
    def _analyze_conversation_history(self, current_text: str, history: List[str]) -> float:
        """
        Analyze if there's a sudden tone shift suggesting sarcasm
        """
        if not history or len(history) < 2:
            return 0.0
        
        score = 0.0
        
        # Get sentiment of recent history
        prev_negative_count = sum(
            1 for msg in history[-3:] 
            if any(neg_word in msg.lower() for neg_word in self.negative_words)
        )
        
        # If previous messages were negative and current is suddenly "positive"
        current_positive = any(pos_word in current_text for pos_word in self.positive_words)
        
        if prev_negative_count >= 2 and current_positive:
            score += 0.3
            logger.debug("Tone shift: negative history → positive statement")
        
        return min(score, 0.3)
    
    def get_sarcasm_type(self, text: str, score: float) -> str:
        """
        Classify the type of sarcasm detected
        """
        text_lower = text.lower()
        
        if score < 0.3:
            return "none"
        
        # Check different types
        if any(phrase in text_lower for phrase in ["oh great", "oh wonderful", "oh perfect"]):
            return "mock_enthusiasm"
        
        if any(word in text_lower for word in ["yeah right", "sure", "of course"]):
            return "disbelief"
        
        if re.search(r"(so|very|really)\s+(helpful|useful)", text_lower):
            return "feigned_gratitude"
        
        if any(word in text_lower for word in ["brilliant", "genius", "smooth"]):
            return "mock_praise"
        
        # Check for passive-aggressive patterns
        if "but" in text_lower or "however" in text_lower:
            return "passive_aggressive"
        
        return "general_sarcasm"