"""
Enhanced Local Coaching Analyzer - WITH ISSUE & RESOLUTION DETECTION
Generates rich, meaningful coaching reports using heuristic analysis
"""
import logging
from typing import Dict, List, Any
from collections import Counter
import re

logger = logging.getLogger(__name__)


class EnhancedLocalAnalyzer:
    """Generate rich coaching insights without Gemini API"""
    
    def __init__(self):
        # Coaching quality indicators
        self.quality_indicators = {
            'powerful_questions': [
                r'\bwhat.*?else\b',
                r'\bhow.*?feel\b',
                r'\bwhat.*?want\b',
                r'\bwhat.*?if\b',
                r'\bwhat.*?stop\b',
                r'\bwhat.*?different\b',
                r'\btell me more\b',
                r'\bhelp me understand\b'
            ],
            'listening': [
                r'\bi hear\b',
                r'\bsounds like\b',
                r'\bso you.*?re saying\b',
                r'\bif i understand\b',
                r'\blet me check\b'
            ],
            'acknowledgment': [
                r'\bi appreciate\b',
                r'\bthat.*?s great\b',
                r'\bwell done\b',
                r'\bexcellent\b'
            ]
        }
    
    def generate_comprehensive_report(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive coaching report"""
        chunks = session_data.get('chunks', [])
        feedback_history = session_data.get('feedback_history', [])
        
        if not chunks:
            return self._empty_report(session_data)
        
        # Analyze conversation
        analysis = self._analyze_conversation(chunks, feedback_history)
        
        # Build comprehensive report
        return {
            'session_id': session_data.get('session_id', 'unknown'),
            'duration_minutes': session_data.get('duration', 0),
            'participants': self._analyze_participants(chunks, feedback_history),
            'grow_phases': self._analyze_grow_distribution(feedback_history),
            'emotional_journey': self._analyze_emotional_journey(chunks, feedback_history),
            'learning_style_analysis': self._analyze_learning_styles(feedback_history),
            'key_insights': self._generate_key_insights(analysis, chunks),
            'coaching_effectiveness': self._analyze_coaching_effectiveness(analysis),
            'recommendations': self._generate_recommendations(analysis),
            'transcript_summary': self._generate_summary(analysis, chunks)
        }
    
    def _analyze_conversation(self, chunks: List, feedback_history: List) -> Dict[str, Any]:
        """Deep analysis of conversation patterns"""
        coach_turns = [c for c in chunks if c.speaker == 'coach']
        coachee_turns = [c for c in chunks if c.speaker == 'coachee']
        
        # Analyze questions
        questions = self._analyze_questions(coach_turns)
        
        # Analyze listening patterns
        listening = self._analyze_listening(coach_turns)
        
        # Extract themes from coachee responses
        themes = self._extract_themes(coachee_turns)
        
        # Analyze engagement patterns
        engagement = self._analyze_engagement_patterns(feedback_history)
        
        # Detect coaching moments
        moments = self._detect_coaching_moments(chunks, feedback_history)
        
        return {
            'questions': questions,
            'listening': listening,
            'themes': themes,
            'engagement': engagement,
            'moments': moments,
            'coach_turns': len(coach_turns),
            'coachee_turns': len(coachee_turns)
        }
    
    def _analyze_questions(self, coach_turns: List) -> Dict[str, Any]:
        """Analyze questioning quality"""
        total_questions = 0
        open_questions = 0
        closed_questions = 0
        powerful_questions = 0
        
        for turn in coach_turns:
            text = turn.transcript.lower()
            
            if '?' in turn.transcript:
                total_questions += 1
                
                if any(word in text for word in ['what', 'how', 'why', 'tell me', 'describe']):
                    open_questions += 1
                else:
                    closed_questions += 1
                
                for pattern in self.quality_indicators['powerful_questions']:
                    if re.search(pattern, text, re.IGNORECASE):
                        powerful_questions += 1
                        break
        
        return {
            'total': total_questions,
            'open': open_questions,
            'closed': closed_questions,
            'powerful': powerful_questions,
            'ratio': open_questions / max(total_questions, 1)
        }
    
    def _analyze_listening(self, coach_turns: List) -> Dict[str, Any]:
        """Analyze active listening indicators"""
        listening_count = 0
        reflection_count = 0
        
        for turn in coach_turns:
            text = turn.transcript.lower()
            
            for pattern in self.quality_indicators['listening']:
                if re.search(pattern, text, re.IGNORECASE):
                    listening_count += 1
                    break
            
            if 'you said' in text or 'you mentioned' in text:
                reflection_count += 1
        
        return {
            'listening_indicators': listening_count,
            'reflections': reflection_count,
            'frequency': listening_count / max(len(coach_turns), 1)
        }
    
    def _extract_themes(self, coachee_turns: List) -> List[str]:
        """Extract main themes from coachee dialogue"""
        theme_keywords = {
            'career': ['job', 'career', 'work', 'promotion', 'company'],
            'goals': ['goal', 'want', 'achieve', 'aspire', 'hope', 'become'],
            'challenges': ['problem', 'difficult', 'struggle', 'challenge', 'issue'],
            'growth': ['learn', 'grow', 'develop', 'improve', 'skill'],
            'relationships': ['team', 'manager', 'colleague', 'people', 'relationship'],
            'confidence': ['confident', 'sure', 'believe', 'doubt', 'afraid'],
            'decision': ['decide', 'choice', 'option', 'choose', 'decision']
        }
        
        theme_counts = Counter()
        
        for turn in coachee_turns:
            text = turn.transcript.lower()
            for theme, keywords in theme_keywords.items():
                if any(kw in text for kw in keywords):
                    theme_counts[theme] += 1
        
        return [theme for theme, _ in theme_counts.most_common(3)]
    
    def _analyze_engagement_patterns(self, feedback_history: List) -> Dict[str, Any]:
        """Analyze engagement patterns over time"""
        if not feedback_history:
            return {'average': 0.5, 'trend': 'stable', 'low_points': 0}
        
        scores = [f.engagement_score for f in feedback_history]
        avg_score = sum(scores) / len(scores)
        
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        avg_first = sum(first_half) / max(len(first_half), 1)
        avg_second = sum(second_half) / max(len(second_half), 1)
        
        if avg_second > avg_first + 0.1:
            trend = 'increasing'
        elif avg_second < avg_first - 0.1:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        low_points = sum(1 for s in scores if s < 0.4)
        
        return {
            'average': avg_score,
            'trend': trend,
            'low_points': low_points,
            'min': min(scores),
            'max': max(scores)
        }
    
    def _detect_coaching_moments(self, chunks: List, feedback_history: List) -> List[str]:
        """Detect notable coaching moments"""
        moments = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.transcript.lower()
            
            if any(word in text for word in ['realize', 'aha', 'understand now', 'makes sense']):
                moments.append(f"Potential breakthrough at {chunk.timestamp:.1f}s: Coachee showed new understanding")
            
            if chunk.speaker == 'coachee' and any(word in text for word in ["but", "can't", "don't know", "impossible"]):
                moments.append(f"Resistance detected at {chunk.timestamp:.1f}s: Limiting belief expressed")
            
            if chunk.speaker == 'coach' and '?' in chunk.transcript:
                for pattern in self.quality_indicators['powerful_questions']:
                    if re.search(pattern, text, re.IGNORECASE):
                        moments.append(f"Powerful question at {chunk.timestamp:.1f}s: '{chunk.transcript[:60]}...'")
                        break
        
        return moments[:5]
    
    def _analyze_participants(self, chunks: List, feedback_history: List) -> Dict[str, Dict]:
        """Analyze participant engagement"""
        coach_chunks = [c for c in chunks if c.speaker == 'coach']
        coachee_chunks = [c for c in chunks if c.speaker == 'coachee']
        
        coach_engagement = [f.engagement_score for f in feedback_history if f.speaker == 'coach']
        coachee_engagement = [f.engagement_score for f in feedback_history if f.speaker == 'coachee']
        
        return {
            'coach': {
                'total_turns': len(coach_chunks),
                'engagement_avg': sum(coach_engagement) / max(len(coach_engagement), 1) if coach_engagement else 0.5,
                'avg_words': sum(len(c.transcript.split()) for c in coach_chunks) / max(len(coach_chunks), 1)
            },
            'coachee': {
                'total_turns': len(coachee_chunks),
                'engagement_avg': sum(coachee_engagement) / max(len(coachee_engagement), 1) if coachee_engagement else 0.5,
                'avg_words': sum(len(c.transcript.split()) for c in coachee_chunks) / max(len(coachee_chunks), 1)
            }
        }
    
    def _analyze_grow_distribution(self, feedback_history: List) -> List[Dict]:
        """Analyze GROW phase distribution"""
        if not feedback_history:
            return []
        
        phase_durations = Counter()
        phase_confidences = {}
        
        for feedback in feedback_history:
            phase = feedback.grow_phase.phase
            phase_durations[phase] += 1
            
            if phase not in phase_confidences:
                phase_confidences[phase] = []
            phase_confidences[phase].append(feedback.grow_phase.confidence)
        
        total = sum(phase_durations.values())
        
        return [
            {
                'phase': phase,
                'percentage': (count / total) * 100,
                'avg_confidence': sum(phase_confidences[phase]) / len(phase_confidences[phase])
            }
            for phase, count in phase_durations.most_common()
        ]
    
    def _analyze_emotional_journey(self, chunks: List, feedback_history: List) -> Dict[str, List]:
        """Analyze emotional journey for each participant"""
        coach_emotions = []
        coachee_emotions = []
        
        for feedback in feedback_history:
            dominant_emotion = max(feedback.emotion_trend.items(), key=lambda x: x[1])
            
            emotion_point = {
                'timestamp': feedback.timestamp,
                'emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1]
            }
            
            if feedback.speaker == 'coach':
                coach_emotions.append(emotion_point)
            else:
                coachee_emotions.append(emotion_point)
        
        return {
            'coach': coach_emotions,
            'coachee': coachee_emotions
        }
    
    def _analyze_learning_styles(self, feedback_history: List) -> Dict[str, float]:
        """Aggregate learning style preferences"""
        if not feedback_history:
            return {'visual': 0.33, 'auditory': 0.33, 'kinesthetic': 0.34}
        
        return {'visual': 0.33, 'auditory': 0.33, 'kinesthetic': 0.34}
    
    def _generate_key_insights(self, analysis: Dict, chunks: List) -> List[str]:
        """Generate key insights from analysis"""
        insights = []
        
        q = analysis['questions']
        if q['total'] > 0:
            insights.append(
                f"Coach asked {q['total']} questions, {q['ratio']:.0%} were open-ended. "
                f"{q['powerful']} powerful questions were used to deepen exploration."
            )
        
        if analysis['themes']:
            themes_str = ', '.join(analysis['themes'])
            insights.append(f"Main coaching themes: {themes_str}")
        
        eng = analysis['engagement']
        insights.append(
            f"Coachee engagement {eng['trend']} throughout session (avg: {eng['average']:.2f}). "
            f"{eng['low_points']} moments of low engagement detected."
        )
        
        listen = analysis['listening']
        if listen['listening_indicators'] > 0:
            insights.append(
                f"Coach demonstrated active listening with {listen['listening_indicators']} listening indicators "
                f"and {listen['reflections']} reflections."
            )
        
        if analysis['moments']:
            insights.append("Notable moments: " + "; ".join(analysis['moments'][:2]))
        
        return insights
    
    def _analyze_coaching_effectiveness(self, analysis: Dict) -> Dict[str, float]:
        """Calculate coaching effectiveness scores"""
        q = analysis['questions']
        questioning_score = min(1.0, q['ratio'] + (q['powerful'] / max(q['total'], 1)) * 0.3)
        
        listen = analysis['listening']
        listening_score = min(1.0, listen['frequency'] * 2)
        
        overall = (questioning_score * 0.5 + listening_score * 0.3 + analysis['engagement']['average'] * 0.2)
        
        return {
            'overall': overall,
            'questioning': questioning_score,
            'listening': listening_score,
            'engagement_management': analysis['engagement']['average']
        }
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        q = analysis['questions']
        if q['ratio'] < 0.6:
            recommendations.append(
                "Increase open-ended questions. Try 'What else?', 'How do you feel about that?', "
                "'What would success look like?'"
            )
        
        if q['powerful'] < 3:
            recommendations.append(
                "Incorporate more powerful questions to deepen thinking: "
                "'What if that weren't true?', 'What's really important here?', 'What are you not saying?'"
            )
        
        listen = analysis['listening']
        if listen['frequency'] < 0.3:
            recommendations.append(
                "Enhance active listening by reflecting back what you hear: "
                "'So what I'm hearing is...', 'It sounds like...', 'Let me check if I understand...'"
            )
        
        eng = analysis['engagement']
        if eng['low_points'] > 2:
            recommendations.append(
                f"Engagement dropped {eng['low_points']} times. When you notice energy decreasing, "
                "check in: 'What's coming up for you right now?', 'Where's your attention?'"
            )
        
        if 'challenges' in analysis['themes']:
            recommendations.append(
                "When discussing challenges, balance problem exploration with solution focus. "
                "Move from 'What's wrong?' to 'What's possible?'"
            )
        
        if q['powerful'] >= 3:
            recommendations.insert(0, f"Great use of powerful questions ({q['powerful']} detected). Keep building on this strength!")
        
        return recommendations[:5]
    
    def _generate_summary(self, analysis: Dict, chunks: List) -> str:
        """Generate summary WITH issue detection and resolution - ENHANCED"""
        themes_str = ', '.join(analysis['themes']) if analysis['themes'] else 'various topics'
        
        q = analysis['questions']
        eng = analysis['engagement']
        
        summary_parts = []
        
        summary_parts.append(
            f"The coach explored {themes_str} with the coachee across {analysis['coach_turns']} coach turns "
            f"and {analysis['coachee_turns']} coachee responses."
        )
        
        # ADDED: Extract issues
        issues_discussed = self._extract_issues_from_conversation(chunks)
        if issues_discussed:
            summary_parts.append(issues_discussed)
        
        # ADDED: Extract approach
        coaching_approach = self._analyze_coaching_approach(chunks, analysis)
        if coaching_approach:
            summary_parts.append(coaching_approach)
        
        if q['total'] > 0:
            summary_parts.append(
                f"Questioning technique showed {q['ratio']:.0%} open-ended questions with "
                f"{q['powerful']} powerful questions that deepened exploration."
            )
        
        summary_parts.append(
            f"Coachee engagement {eng['trend']} throughout the session, "
            f"averaging {eng['average']:.2f} with {eng['low_points']} moments requiring re-engagement."
        )
        
        if analysis['moments']:
            summary_parts.append(f"Key moment: {analysis['moments'][0]}")
        
        return ' '.join(summary_parts)
    
    def _extract_issues_from_conversation(self, chunks: List) -> str:
        """Extract specific issues mentioned - NEW METHOD"""
        coachee_chunks = [c for c in chunks if c.speaker == 'coachee']
        
        issue_keywords = {
            'goal': ['want to', 'goal', 'achieve', 'become', 'aspire', 'best'],
            'emotion': ['feel', 'sad', 'happy', 'worried', 'anxious', 'excited', 'ecstatic'],
            'challenge': ['difficult', 'hard', 'struggling', 'boring', 'problem'],
            'obstacle': ["can't", "don't know", 'unable', 'impossible'],
        }
        
        detected_issues = {}
        
        for chunk in coachee_chunks:
            text = chunk.transcript.lower()
            
            for issue_type, keywords in issue_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        if issue_type not in detected_issues:
                            detected_issues[issue_type] = []
                        
                        sentences = chunk.transcript.split('.')
                        for sentence in sentences:
                            if keyword in sentence.lower():
                                detected_issues[issue_type].append(sentence.strip())
                                break
                        break
        
        issue_descriptions = []
        
        if 'goal' in detected_issues:
            goal_text = detected_issues['goal'][0][:80]
            issue_descriptions.append(f"expressed goal: '{goal_text}'")
        
        if 'emotion' in detected_issues:
            emotion_text = detected_issues['emotion'][0][:80]
            issue_descriptions.append(f"shared emotional state: '{emotion_text}'")
        
        if 'challenge' in detected_issues:
            challenge_text = detected_issues['challenge'][0][:80]
            issue_descriptions.append(f"mentioned challenge: '{challenge_text}'")
        
        if issue_descriptions:
            return "Coachee " + "; ".join(issue_descriptions[:2])
        
        return ""
    
    def _analyze_coaching_approach(self, chunks: List, analysis: Dict) -> str:
        """Analyze how coach tackled issues - NEW METHOD"""
        coach_chunks = [c for c in chunks if c.speaker == 'coach']
        
        approaches = []
        
        if analysis['questions']['powerful'] >= 2:
            approaches.append("used powerful questions")
        
        if analysis['listening']['listening_indicators'] >= 2:
            approaches.append("demonstrated active listening")
        
        techniques_used = set()
        
        for chunk in coach_chunks:
            text = chunk.transcript.lower()
            
            if any(phrase in text for phrase in ['what if', 'another way']):
                techniques_used.add('reframing')
            
            if any(phrase in text for phrase in ['i hear', 'i understand', 'appreciate']):
                techniques_used.add('acknowledgment')
            
            if any(phrase in text for phrase in ['what stops', 'what if you']):
                techniques_used.add('challenging')
            
            if any(phrase in text for phrase in ['what do you want', 'your goal']):
                techniques_used.add('goal clarification')
        
        if techniques_used:
            approaches.append(f"employed {', '.join(list(techniques_used)[:2])}")
        
        if approaches:
            return "Coach " + "; ".join(approaches[:2])
        
        return ""
    
    def _empty_report(self, session_data: Dict) -> Dict[str, Any]:
        """Return empty report structure"""
        return {
            'session_id': session_data.get('session_id', 'unknown'),
            'duration_minutes': 0,
            'participants': {},
            'grow_phases': [],
            'emotional_journey': {'coach': [], 'coachee': []},
            'learning_style_analysis': {},
            'key_insights': ['No conversation data available'],
            'coaching_effectiveness': {'overall': 0.0, 'questioning': 0.0, 'listening': 0.0},
            'recommendations': ['Record a coaching session to receive insights'],
            'transcript_summary': 'No data recorded'
        }