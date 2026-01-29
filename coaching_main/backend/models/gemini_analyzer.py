import json
import re
import logging
from typing import Dict, Any
import google.generativeai as genai

from backend.schemas.data_models import (
    AudioChunk,
    ModelInferences,
    RealTimeFeedback,
    SessionReport,
)

logger = logging.getLogger(__name__)


class GeminiAnalyzer:
    """Handles AI-powered analysis using Google Gemini"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Use the updated model name (gemini-1.5-pro or gemini-1.5-flash)
        self.model = genai.GenerativeModel('gemini-2.5-flash')  # Changed from gemini-pro
        
    def _parse_gemini_json(self, raw_text: str) -> dict:
        """
        Parse JSON from Gemini response, handling markdown code blocks.
        
        Args:
            raw_text: Raw text response from Gemini
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If JSON cannot be parsed
        """
        text = raw_text.strip()
        
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Extract JSON from markdown code blocks (```json ... ```)
        json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(json_block_pattern, text, re.DOTALL)
        
        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON between curly braces
        brace_pattern = r'\{.*\}'
        brace_matches = re.findall(brace_pattern, text, re.DOTALL)
        
        if brace_matches:
            for match in sorted(brace_matches, key=len, reverse=True):
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        logger.error(f"Failed to parse JSON. Raw output: {text[:500]}...")
        raise ValueError(f"Could not extract valid JSON from response")

    async def analyze_real_time(
        self,
        chunk: AudioChunk,
        inferences: ModelInferences,
        context: Dict[str, Any],
    ) -> RealTimeFeedback:
        """Generate real-time feedback for a coaching interaction"""
        
        prompt = f"""Analyze this coaching interaction and provide real-time feedback in JSON format.

Current interaction:
- Speaker: {chunk.speaker}
- Text: "{chunk.transcript}"
- Emotion: {inferences.emotion}
- Engagement: {inferences.interest_level}
- Sarcasm detected: {inferences.sarcasm_detected}

Session context:
- Previous interactions: {len(context.get('previous_chunks', []))}
- Session duration: {context.get('duration_minutes', 0):.1f} minutes

Provide feedback as JSON with this structure:
{{
    "engagement_score": 0.0-1.0,
    "emotion": "string",
    "coaching_technique": "string",
    "grow_phase": "Goal/Reality/Options/Way Forward",
    "learning_style": "Visual/Auditory/Kinesthetic",
    "suggestion": "brief actionable suggestion",
    "highlight": "key moment or insight"
}}

Return ONLY the JSON, no markdown formatting."""

        try:
            response = await self.model.generate_content_async(prompt)
            feedback_dict = self._parse_gemini_json(response.text)
            
            return RealTimeFeedback(
                timestamp=chunk.timestamp,
                speaker=chunk.speaker,
                engagement_score=feedback_dict.get('engagement_score', 0.5),
                emotion=feedback_dict.get('emotion', inferences.emotion),
                coaching_technique=feedback_dict.get('coaching_technique', 'Unknown'),
                grow_phase=feedback_dict.get('grow_phase', 'Reality'),
                learning_style=feedback_dict.get('learning_style', 'Unknown'),
                suggestion=feedback_dict.get('suggestion', 'Continue with current approach'),
                highlight=feedback_dict.get('highlight', ''),
            )
            
        except Exception as e:
            logger.error(f"Real-time analysis error: {e}")
            # Return fallback feedback
            return RealTimeFeedback(
                timestamp=chunk.timestamp,
                speaker=chunk.speaker,
                engagement_score=0.5,
                emotion=inferences.emotion,
                coaching_technique='Active Listening',
                grow_phase='Reality',
                learning_style='Unknown',
                suggestion='Continue the conversation',
                highlight='',
            )

    async def generate_session_report(self, session_data: Dict[str, Any]) -> SessionReport:
        """Generate comprehensive session report"""
        
        prompt = f"""Analyze this coaching session and generate a comprehensive report in JSON format.

Session data:
- Session ID: {session_data.get('session_id')}
- Duration: {session_data.get('duration', 0):.1f} minutes
- Total interactions: {len(session_data.get('chunks', []))}
- Average engagement: {self._calculate_avg_engagement(session_data.get('chunks', []))}

Interactions summary:
{self._format_chunks_for_prompt(session_data.get('chunks', [])[:10])}

Generate a report with this EXACT JSON structure (no markdown, no code blocks):
{{
    "session_id": "string",
    "duration_minutes": 0.0,
    "participants": {{
        "coach": {{"engagement_avg": 0.0}},
        "coachee": {{"engagement_avg": 0.0}}
    }},
    "grow_phases": [],
    "emotional_journey": {{
        "coach": [],
        "coachee": []
    }},
    "learning_style_analysis": {{}},
    "key_insights": ["insight1", "insight2"],
    "coaching_effectiveness": {{
        "overall": 0.0,
        "questioning": 0.0,
        "listening": 0.0
    }},
    "recommendations": ["rec1", "rec2"],
    "transcript_summary": "string"
}}

Return ONLY valid JSON without any markdown formatting or code blocks."""

        try:
            response = await self.model.generate_content_async(prompt)
            
            # Parse the JSON response
            report_dict = self._parse_gemini_json(response.text)
            
            # Create SessionReport from parsed dict
            return SessionReport(
                session_id=report_dict.get('session_id', session_data.get('session_id', 'unknown')),
                duration_minutes=report_dict.get('duration_minutes', session_data.get('duration', 0)),
                participants=report_dict.get('participants', {
                    "coach": {"engagement_avg": 0.5},
                    "coachee": {"engagement_avg": 0.5}
                }),
                grow_phases=report_dict.get('grow_phases', []),
                emotional_journey=report_dict.get('emotional_journey', {"coach": [], "coachee": []}),
                learning_style_analysis=report_dict.get('learning_style_analysis', {}),
                key_insights=report_dict.get('key_insights', ["Session completed"]),
                coaching_effectiveness=report_dict.get('coaching_effectiveness', {
                    "overall": 0.5,
                    "questioning": 0.5,
                    "listening": 0.5
                }),
                recommendations=report_dict.get('recommendations', ["Continue coaching practices"]),
                transcript_summary=report_dict.get('transcript_summary', "Session summary unavailable")
            )
            
        except ValueError as e:
            logger.error(f"Failed to parse Gemini JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Gemini report generation error: {e}")
            raise

    def _calculate_avg_engagement(self, chunks: list) -> float:
        """Calculate average engagement from chunks"""
        if not chunks:
            return 0.0
        
        engagements = [c.get('engagement', 0.5) for c in chunks if isinstance(c, dict)]
        return sum(engagements) / len(engagements) if engagements else 0.5

    def _format_chunks_for_prompt(self, chunks: list, max_chunks: int = 10) -> str:
        """Format chunks for inclusion in prompt"""
        if not chunks:
            return "No interactions recorded"
        
        formatted = []
        for i, chunk in enumerate(chunks[:max_chunks]):
            if isinstance(chunk, dict):
                speaker = chunk.get('speaker', 'unknown')
                text = chunk.get('transcript', '')
                formatted.append(f"{i+1}. [{speaker}]: {text[:100]}")
        
        return "\n".join(formatted) if formatted else "No interactions recorded"