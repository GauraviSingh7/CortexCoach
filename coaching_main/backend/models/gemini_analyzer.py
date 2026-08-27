"""
Gemini narrative layer.

Gemini is used to *write* about the session, not to measure it. It receives
the metrics the pipeline already computed - GROW distribution and coverage,
engagement, listening and questioning breakdowns, sarcasm and digression
moments, learning style, and which models were degraded - and returns prose
only. The caller merges just the prose fields back in.

Previously this class received nothing but ``speaker``/``transcript``/
``timestamp`` for the first ten chunks and was asked to invent
``grow_phases``, ``coaching_effectiveness``, ``emotional_journey`` and
engagement figures. Those invented numbers then replaced the computed ones,
which is why a Gemini-generated report could contain less information than
the local analyzer's.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiAnalyzer:
    """Generates the narrative sections of a session report."""

    # Requires google-generativeai>=0.8.0 for `gemini-2.5-flash`.
    _MODEL_CANDIDATES = (
        "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro",
    )

    def __init__(self, api_key: str):
        self.model = None
        try:
            genai.configure(api_key=api_key)
        except Exception as exc:
            logger.warning("Gemini configure() failed: %s. Gemini disabled.", exc)
            return

        for candidate in self._MODEL_CANDIDATES:
            try:
                self.model = genai.GenerativeModel(candidate)
                logger.info("Gemini model initialised: %s", candidate)
                break
            except Exception as exc:
                logger.debug("Gemini model '%s' unavailable (%s)", candidate, exc)

        if not self.model:
            logger.warning("No compatible Gemini model found; local reports only.")

    # -- report narration --------------------------------------------------

    async def generate_session_report(
        self,
        session_data: Dict[str, Any],
        computed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return narrative fields for a session.

        Only ``key_insights``, ``recommendations`` and ``transcript_summary``
        are returned; the caller keeps its own computed metrics.
        """
        if not self.model:
            raise RuntimeError("Gemini model not available")

        computed = computed or {}
        prompt = self._build_prompt(session_data, computed)
        response = await self.model.generate_content_async(prompt)
        parsed = self._parse_json(response.text)

        return {
            "key_insights": self._as_str_list(parsed.get("key_insights")),
            "recommendations": self._as_str_list(parsed.get("recommendations")),
            "transcript_summary": str(parsed.get("transcript_summary") or "").strip(),
        }

    def _build_prompt(
        self, session_data: Dict[str, Any], computed: Dict[str, Any]
    ) -> str:
        chunks = session_data.get("chunks", []) or []
        model_status = computed.get("model_status", {}) or {}
        degraded = model_status.get("degraded", [])

        caveat = (
            "\nIMPORTANT: these signals came from rule-based heuristics rather "
            f"than trained models: {', '.join(degraded)}. Do not overstate "
            "their precision in your narrative.\n" if degraded else ""
        )

        return f"""You are an expert coaching supervisor writing up a session.

All figures below were computed by the application. Treat them as facts:
do NOT recalculate, contradict or invent metrics.

SESSION
- Duration: {session_data.get('duration', 0):.1f} minutes
- Turns: {len(chunks)} ({computed.get('participants', {})})

GROW PHASES
{self._fmt(computed.get('grow_phases'))}
Coverage: {self._fmt(computed.get('grow_coverage'))}

COACHING EFFECTIVENESS
{self._fmt(computed.get('coaching_effectiveness'))}

LEARNING STYLE (VAK)
{self._fmt(computed.get('learning_style_analysis')) or 'Insufficient data'}

SARCASM
{self._fmt(computed.get('sarcasm_summary')) or 'None detected'}

DIGRESSION
{self._fmt(computed.get('digression_summary')) or 'None detected'}

EMOTIONAL JOURNEY
{self._fmt(computed.get('emotional_journey'))}
{caveat}
TRANSCRIPT
{self._format_transcript(chunks)}

Write the narrative sections. Return ONLY this JSON, no markdown fences:
{{
  "key_insights": ["3-6 specific observations grounded in the data above"],
  "recommendations": ["3-5 concrete, actionable suggestions for the coach"],
  "transcript_summary": "One paragraph: what was discussed, how the coach worked, where it landed."
}}"""

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _format_transcript(chunks: Sequence[Any], limit: int = 60) -> str:
        """Full conversation, not the first ten turns."""
        if not chunks:
            return "No interactions recorded."
        lines = []
        for index, chunk in enumerate(chunks[:limit], start=1):
            speaker = getattr(chunk, "speaker", None) or (
                chunk.get("speaker") if isinstance(chunk, dict) else "unknown"
            )
            text = getattr(chunk, "transcript", None) or (
                chunk.get("transcript") if isinstance(chunk, dict) else ""
            )
            lines.append(f"{index}. [{speaker}] {text}")
        if len(chunks) > limit:
            lines.append(f"... ({len(chunks) - limit} further turns omitted)")
        return "\n".join(lines)

    @staticmethod
    def _fmt(value: Any) -> str:
        if not value:
            return ""
        return json.dumps(value, indent=2, default=str)

    @staticmethod
    def _as_str_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return []

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Extract a JSON object from a model response."""
        text = (raw_text or "").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for match in re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL):
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        braces = re.findall(r"\{.*\}", text, re.DOTALL)
        for match in sorted(braces, key=len, reverse=True):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        logger.error("Could not parse Gemini JSON. Raw output: %.500s", text)
        raise ValueError("Could not extract valid JSON from Gemini response")
