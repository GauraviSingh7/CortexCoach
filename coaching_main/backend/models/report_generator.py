"""
Session Report Generator
Uses Google Gemini to analyze session data and generate comprehensive reports
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import google.generativeai as genai

from backend.schemas.data_models import SessionReport

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive session reports using AI analysis"""

    def __init__(self, gemini_key: str):
        self.gemini_key = gemini_key
        
        # Configure Gemini
        if gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("✅ Gemini configured for report generation")
            except Exception as e:
                logger.warning(f"Failed to configure Gemini: {e}")
                self.model = None
        else:
            logger.warning("No Gemini API key provided - reports will be basic")
            self.model = None

    async def generate_report(self, session_data: Dict[str, Any]) -> SessionReport:
        """Generate comprehensive session report"""
        
        logger.info("📊 Generating session report...")
        
        try:
            # Calculate session duration
            start_time = session_data.get("start_time", datetime.now())
            duration_minutes = (datetime.now() - start_time).total_seconds() / 60
            
            # Extract chunks and feedback
            chunks = session_data.get("chunks", [])
            feedback_history = session_data.get("feedback_history", [])
            
            # Separate coach and coachee data
            coach_feedback = [f for f in feedback_history if f.speaker == "coach"]
            coachee_feedback = [f for f in feedback_history if f.speaker == "coachee"]
            
            # Calculate participant metrics
            participants = {
                "coach": self._calculate_participant_metrics(coach_feedback),
                "coachee": self._calculate_participant_metrics(coachee_feedback)
            }
            
            # Extract GROW phases
            grow_phases = self._extract_grow_phases(feedback_history)
            
            # Analyze emotional journey
            emotional_journey = self._analyze_emotional_journey(feedback_history)
            
            # Analyze learning style
            learning_style_analysis = self._analyze_learning_style(feedback_history)
            
            # Calculate coaching effectiveness
            coaching_effectiveness = self._calculate_coaching_effectiveness(coach_feedback)
            
            # Generate insights and recommendations
            if self.model and len(chunks) > 0:
                key_insights, recommendations, transcript_summary = await self._generate_ai_insights(
                    chunks, feedback_history
                )
            else:
                key_insights = self._generate_basic_insights(feedback_history)
                recommendations = self._generate_basic_recommendations(feedback_history)
                transcript_summary = self._generate_basic_summary(chunks)
            
            # Create report
            report = SessionReport(
                session_id=session_data.get("session_id", "unknown"),
                duration_minutes=duration_minutes,
                participants=participants,
                grow_phases=grow_phases,
                emotional_journey=emotional_journey,
                learning_style_analysis=learning_style_analysis,
                key_insights=key_insights,
                coaching_effectiveness=coaching_effectiveness,
                recommendations=recommendations,
                transcript_summary=transcript_summary
            )
            
            logger.info("✅ Report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            # Return basic report on error
            return self._generate_fallback_report(session_data)

    def _calculate_participant_metrics(self, feedback_list: List) -> Dict[str, float]:
        """Calculate metrics for a participant"""
        if not feedback_list:
            return {
                "avg_engagement": 0.0,
                "total_turns": 0,
                "avg_quality": 0.0
            }
        
        total_engagement = sum(f.engagement_score for f in feedback_list)
        avg_engagement = total_engagement / len(feedback_list)
        
        total_quality = sum(f.coaching_quality.get("overall", 0.5) for f in feedback_list)
        avg_quality = total_quality / len(feedback_list)
        
        return {
            "avg_engagement": avg_engagement,
            "total_turns": len(feedback_list),
            "avg_quality": avg_quality
        }

    def _extract_grow_phases(self, feedback_history: List) -> List[Dict]:
        """Extract GROW phase progression"""
        phases = []
        
        for feedback in feedback_history:
            if feedback.speaker == "coach":  # Only track coach's GROW phases
                phases.append({
                    "timestamp": feedback.timestamp,
                    "phase": feedback.grow_phase.phase,
                    "confidence": feedback.grow_phase.confidence,
                    "reasoning": feedback.grow_phase.reasoning
                })
        
        return phases

    def _analyze_emotional_journey(self, feedback_history: List) -> Dict[str, List]:
        """Analyze emotional progression throughout session"""
        journey = {
            "coach": [],
            "coachee": []
        }
        
        for feedback in feedback_history:
            emotions = feedback.emotion_trend
            journey[feedback.speaker].append({
                "timestamp": feedback.timestamp,
                "emotions": emotions,
                "dominant": max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
            })
        
        return journey

    def _analyze_learning_style(self, feedback_history: List) -> Dict[str, float]:
        """Analyze predominant learning style (VAK)"""
        # Aggregate VAK scores from coachee feedback
        vak_totals = {"visual": 0.0, "auditory": 0.0, "kinesthetic": 0.0}
        count = 0
        
        for feedback in feedback_history:
            if feedback.speaker == "coachee":
                # Note: VAK data would come from ModelInferences if available
                # For now, return balanced distribution
                count += 1
        
        if count > 0:
            return {"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34}
        else:
            return {"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34}

    def _calculate_coaching_effectiveness(self, coach_feedback: List) -> Dict[str, float]:
        """Calculate overall coaching effectiveness metrics"""
        if not coach_feedback:
            return {
                "overall": 0.5,
                "questioning": 0.5,
                "listening": 0.5,
                "grow_alignment": 0.5
            }
        
        total_quality = sum(f.coaching_quality.get("overall", 0.5) for f in coach_feedback)
        total_questioning = sum(f.coaching_quality.get("questioning", 0.5) for f in coach_feedback)
        total_listening = sum(f.coaching_quality.get("listening", 0.5) for f in coach_feedback)
        
        count = len(coach_feedback)
        
        # Calculate GROW alignment (how well phases were followed)
        phase_sequence = [f.grow_phase.phase for f in coach_feedback]
        grow_alignment = self._calculate_grow_alignment(phase_sequence)
        
        return {
            "overall": total_quality / count,
            "questioning": total_questioning / count,
            "listening": total_listening / count,
            "grow_alignment": grow_alignment
        }

    def _calculate_grow_alignment(self, phase_sequence: List[str]) -> float:
        """Calculate how well the GROW model was followed"""
        if not phase_sequence:
            return 0.5
        
        # Ideal progression: Goal -> Reality -> Options -> Way Forward
        ideal_order = ["Goal", "Reality", "Options", "Way Forward"]
        
        # Simple scoring: check if phases appear in roughly the right order
        score = 0.5  # Base score
        
        # Check if Goal appears early
        if "Goal" in phase_sequence[:len(phase_sequence)//2]:
            score += 0.1
        
        # Check if Way Forward appears late
        if "Way Forward" in phase_sequence[len(phase_sequence)//2:]:
            score += 0.1
        
        return min(score, 1.0)

    async def _generate_ai_insights(self, chunks: List, feedback_history: List) -> tuple:
        """Generate insights using Gemini AI"""
        try:
            # Create transcript summary
            transcript = "\n".join([
                f"{chunk.speaker}: {chunk.transcript}"
                for chunk in chunks[:50]  # Limit to first 50 for API limits
            ])
            
            prompt = f"""
            Analyze this coaching session transcript and provide:
            
            1. Three key insights about the coaching session
            2. Three specific recommendations for improvement
            3. A brief summary of the session (2-3 sentences)
            
            Transcript:
            {transcript}
            
            Format your response as:
            INSIGHTS:
            1. [insight]
            2. [insight]
            3. [insight]
            
            RECOMMENDATIONS:
            1. [recommendation]
            2. [recommendation]
            3. [recommendation]
            
            SUMMARY:
            [summary]
            """
            
            response = self.model.generate_content(prompt)
            text = response.text
            
            # Parse response
            insights = []
            recommendations = []
            summary = ""
            
            current_section = None
            for line in text.split("\n"):
                line = line.strip()
                if "INSIGHTS:" in line:
                    current_section = "insights"
                elif "RECOMMENDATIONS:" in line:
                    current_section = "recommendations"
                elif "SUMMARY:" in line:
                    current_section = "summary"
                elif line and current_section:
                    if current_section == "insights" and line[0].isdigit():
                        insights.append(line[3:].strip())
                    elif current_section == "recommendations" and line[0].isdigit():
                        recommendations.append(line[3:].strip())
                    elif current_section == "summary":
                        summary += line + " "
            
            return insights or ["Session analyzed"], recommendations or ["Continue coaching"], summary.strip() or "Coaching session completed"
            
        except Exception as e:
            logger.error(f"AI insights error: {e}")
            return (
                self._generate_basic_insights(feedback_history),
                self._generate_basic_recommendations(feedback_history),
                self._generate_basic_summary(chunks)
            )

    def _generate_basic_insights(self, feedback_history: List) -> List[str]:
        """Generate basic insights without AI"""
        insights = []
        
        if feedback_history:
            avg_engagement = sum(f.engagement_score for f in feedback_history) / len(feedback_history)
            insights.append(f"Average engagement level was {avg_engagement:.2f}")
            
            coach_count = sum(1 for f in feedback_history if f.speaker == "coach")
            coachee_count = len(feedback_history) - coach_count
            insights.append(f"Session had {coach_count} coach turns and {coachee_count} coachee turns")
            
            phases = [f.grow_phase.phase for f in feedback_history if f.speaker == "coach"]
            if phases:
                most_common = max(set(phases), key=phases.count)
                insights.append(f"Most time spent in '{most_common}' phase of GROW model")
        else:
            insights.append("Session data is limited")
        
        return insights

    def _generate_basic_recommendations(self, feedback_history: List) -> List[str]:
        """Generate basic recommendations without AI"""
        recommendations = []
        
        if feedback_history:
            avg_engagement = sum(f.engagement_score for f in feedback_history) / len(feedback_history)
            
            if avg_engagement < 0.5:
                recommendations.append("Focus on improving engagement through more open-ended questions")
            
            coach_feedback = [f for f in feedback_history if f.speaker == "coach"]
            if coach_feedback:
                avg_quality = sum(f.coaching_quality.get("overall", 0.5) for f in coach_feedback) / len(coach_feedback)
                if avg_quality < 0.7:
                    recommendations.append("Work on active listening and questioning techniques")
            
            recommendations.append("Continue regular coaching sessions for sustained improvement")
        else:
            recommendations.append("Complete longer sessions for better insights")
        
        return recommendations

    def _generate_basic_summary(self, chunks: List) -> str:
        """Generate basic summary without AI"""
        if not chunks:
            return "No session data available"
        
        duration = len(chunks)
        coach_count = sum(1 for c in chunks if c.speaker == "coach")
        
        return f"Coaching session with {duration} interactions, {coach_count} from coach. Session covered various topics with moderate engagement."

    def _generate_fallback_report(self, session_data: Dict[str, Any]) -> SessionReport:
        """Generate minimal fallback report on error"""
        return SessionReport(
            session_id=session_data.get("session_id", "unknown"),
            duration_minutes=0.0,
            participants={
                "coach": {"avg_engagement": 0.0, "total_turns": 0, "avg_quality": 0.0},
                "coachee": {"avg_engagement": 0.0, "total_turns": 0, "avg_quality": 0.0}
            },
            grow_phases=[],
            emotional_journey={"coach": [], "coachee": []},
            learning_style_analysis={"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34},
            key_insights=["Report generation failed - data may be incomplete"],
            coaching_effectiveness={
                "overall": 0.5,
                "questioning": 0.5,
                "listening": 0.5,
                "grow_alignment": 0.5
            },
            recommendations=["Retry session with proper configuration"],
            transcript_summary="Session data unavailable"
        )