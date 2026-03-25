"""
Contextual Real-time Suggestion Engine
Generates specific, actionable coaching suggestions based on conversation context
"""
import logging
from typing import List, Dict
from backend.schemas.data_models import AudioChunk, ModelInferences, GROWPhase

logger = logging.getLogger(__name__)


class ContextualSuggestionEngine:
    """Generate contextual, specific coaching suggestions in real-time"""
    
    def __init__(self):
        # Pattern libraries for different scenarios
        self.coachee_signals = {
            'uncertainty': ['don\'t know', 'not sure', 'confused', 'unclear', 'maybe'],
            'resistance': ['but', 'can\'t', 'won\'t', 'impossible', 'never', 'always'],
            'breakthrough': ['realize', 'aha', 'understand now', 'makes sense', 'i see'],
            'emotion': ['feel', 'worried', 'anxious', 'excited', 'frustrated', 'angry', 'happy'],
            'commitment': ['will', 'going to', 'plan to', 'commit', 'promise'],
            'exploration': ['could', 'might', 'what if', 'wondering', 'thinking about']
        }
        
        self.coach_patterns = {
            'closed_question': r'^(is|are|do|does|did|can|will|would|should|have|has)\s',
            'telling': ['you should', 'you need to', 'you have to', 'you must'],
            'powerful_question': ['what else', 'what if', 'how would', 'what would', 'tell me more'],
            'reflection': ['sounds like', 'i hear', 'so you', 'what i\'m hearing']
        }
    
    def generate_suggestions(
        self, 
        chunk: AudioChunk, 
        inferences: ModelInferences,
        grow_phase: GROWPhase,
        conversation_history: List[AudioChunk]
    ) -> List[str]:
        """Generate 5-7 contextual suggestions"""
        suggestions = []
        
        if chunk.speaker == 'coach':
            suggestions = self._generate_coach_suggestions(
                chunk, inferences, grow_phase, conversation_history
            )
        else:
            suggestions = self._generate_coachee_suggestions(
                chunk, inferences, grow_phase, conversation_history
            )
        
        # Ensure we have 5-7 suggestions
        if len(suggestions) < 5:
            suggestions.extend(self._get_general_suggestions(grow_phase)[:7-len(suggestions)])
        
        return suggestions[:7]
    
    def _generate_coach_suggestions(
        self, 
        chunk: AudioChunk,
        inferences: ModelInferences,
        grow_phase: GROWPhase,
        history: List[AudioChunk]
    ) -> List[str]:
        """Generate suggestions for coach turns"""
        suggestions = []
        text_lower = chunk.transcript.lower()
        
        # Check for closed questions
        if '?' in chunk.transcript:
            is_closed = any(chunk.transcript.lower().startswith(prefix) 
                          for prefix in ['is ', 'are ', 'do ', 'does ', 'did ', 'can ', 'will '])
            
            if is_closed:
                suggestions.append(
                    "🔄 That was a closed question. Try rephrasing: "
                    "'What are your thoughts on...?' or 'How do you see...?'"
                )
            else:
                suggestions.append("✅ Great open-ended question! Now give space to explore the answer fully.")
        
        # Check for telling vs asking
        if any(phrase in text_lower for phrase in self.coach_patterns['telling']):
            suggestions.append(
                "⚠️ Noticed advice-giving. Consider asking: 'What options do you see?' "
                "to keep ownership with the coachee."
            )
        
        # Check for powerful questions
        has_powerful = any(phrase in text_lower for phrase in self.coach_patterns['powerful_question'])
        if has_powerful:
            suggestions.append("💎 Powerful question! This deepens exploration. Follow up with silence to let them think.")
        
        # Phase-specific suggestions
        phase_suggestions = self._get_phase_specific_suggestions(grow_phase, chunk.speaker)
        suggestions.extend(phase_suggestions[:2])
        
        # Engagement-based suggestions
        if inferences.interest_level < 0.4:
            suggestions.append(
                "📉 Energy seems low. Check in: 'Where are you right now?' or "
                "'What's coming up for you as we talk about this?'"
            )
        
        # Check conversation balance
        recent_coach_turns = sum(1 for c in history[-5:] if c.speaker == 'coach')
        if recent_coach_turns >= 4:
            suggestions.append(
                "⚖️ You've been talking a lot. Create space: Ask a question and pause for 5+ seconds."
            )
        
        return suggestions
    
    def _generate_coachee_suggestions(
        self,
        chunk: AudioChunk,
        inferences: ModelInferences,
        grow_phase: GROWPhase,
        history: List[AudioChunk]
    ) -> List[str]:
        """Generate suggestions based on coachee responses"""
        suggestions = []
        text_lower = chunk.transcript.lower()
        
        # Detect uncertainty
        if any(signal in text_lower for signal in self.coachee_signals['uncertainty']):
            suggestions.append(
                "🤔 Coachee expressed uncertainty. Help clarify: "
                "'What part feels unclear?' or 'What would help you know?'"
            )
        
        # Detect resistance/limiting beliefs
        if any(signal in text_lower for signal in self.coachee_signals['resistance']):
            suggestions.append(
                "🚧 Limiting belief detected. Gently challenge: "
                "'What if that weren't true?' or 'What makes that impossible?'"
            )
        
        # Detect breakthrough moments
        if any(signal in text_lower for signal in self.coachee_signals['breakthrough']):
            suggestions.append(
                "💡 Breakthrough moment! Reinforce it: 'What's significant about that realization?' "
                "and give them time to process."
            )
        
        # Detect emotional content
        if any(signal in text_lower for signal in self.coachee_signals['emotion']):
            suggestions.append(
                "❤️ Emotion present. Acknowledge it: 'I hear this brings up feelings. "
                "What's that telling you?'"
            )
        
        # Detect commitment language
        if any(signal in text_lower for signal in self.coachee_signals['commitment']):
            suggestions.append(
                "🎯 Commitment language! Strengthen it: 'What specific first step will you take?' "
                "or 'When exactly will you do this?'"
            )
        
        # Engagement-based
        if inferences.interest_level > 0.7:
            suggestions.append(
                "⚡ High engagement! Coachee is ready to go deeper. "
                "Ask: 'What else?' or 'What are you not saying?'"
            )
        elif inferences.interest_level < 0.4:
            suggestions.append(
                "📊 Engagement dropped. Pause and check in: "
                "'You seem quieter. What's happening for you right now?'"
            )
        
        # Response length
        word_count = len(chunk.transcript.split())
        if word_count < 5:
            suggestions.append(
                "🔇 Brief response. Explore further: 'Tell me more about that' or "
                "'What else is important here?'"
            )
        elif word_count > 50:
            suggestions.append(
                "🗣️ Coachee is opening up! When they finish, reflect back what you heard "
                "to show understanding."
            )
        
        return suggestions
    
    def _get_phase_specific_suggestions(self, grow_phase: GROWPhase, speaker: str) -> List[str]:
        """Get suggestions specific to GROW phase"""
        phase = grow_phase.phase
        suggestions = []
        
        if speaker == 'coach':
            if phase == "Goal":
                suggestions.append(
                    "🎯 Goal Phase: Clarify the outcome. Ask: 'What would success look like?' "
                    "or 'How will you know you've achieved this?'"
                )
                if grow_phase.confidence > 0.7:
                    suggestions.append(
                        "✅ Goal seems clear. Consider moving to Reality: "
                        "'Where are you now in relation to this goal?'"
                    )
            
            elif phase == "Reality":
                suggestions.append(
                    "🔍 Reality Phase: Explore current state. Try: 'What's happening right now?' "
                    "or 'What have you tried so far?'"
                )
                suggestions.append(
                    "💭 In Reality, help them see patterns: 'What do you notice?' "
                    "or 'What's getting in the way?'"
                )
            
            elif phase == "Options":
                suggestions.append(
                    "💡 Options Phase: Generate possibilities. Ask: 'What could you do?' "
                    "or 'If anything were possible, what would you try?'"
                )
                suggestions.append(
                    "🌈 Expand options: 'What else?' Keep asking until they have 5+ options."
                )
            
            elif phase == "Way Forward":
                suggestions.append(
                    "🚀 Way Forward: Lock in commitment. Ask: 'Which option will you choose?' "
                    "and 'What's your first step?'"
                )
                suggestions.append(
                    "📅 Make it concrete: 'When exactly will you do this?' "
                    "and 'What might get in the way?'"
                )
        
        return suggestions
    
    def _get_general_suggestions(self, grow_phase: GROWPhase) -> List[str]:
        """Get general fallback suggestions"""
        return [
            "Use silence - count to 5 after they finish to let them think deeper",
            "Ask 'What else?' to explore beyond the first answer",
            "Reflect back what you hear to show understanding",
            "Notice their energy level and name what you observe",
            "Stay curious - avoid advice unless they explicitly request it",
            "Check assumptions: 'What makes you say that?'",
            "Build on their language - use their words, not yours"
        ]
    
    def get_suggestion_for_silence(self, history: List[AudioChunk]) -> str:
        """Special suggestion when there's a long pause"""
        if not history:
            return "Take a deep breath and start with: 'What would you like to explore today?'"
        
        last_chunk = history[-1]
        if last_chunk.speaker == 'coachee':
            return "Give them space to think. Silence is powerful. Count to 5 before speaking."
        else:
            return "You asked a question - now wait. Let the coachee process and respond fully."