"""Real-time coaching suggestions for the observer dashboard."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_MAX_SUGGESTIONS = 7

_SARCASM_PROMPTS = {
    "passive_aggressive": (
        "Passive-aggressive language detected. Explore what is really "
        "bothering them: 'What's frustrating you about this?'"
    ),
    "ironic_commentary": (
        "Irony detected - often a cover for frustration. Name it gently: "
        "'There's something under that. What's the frustration?'"
    ),
    "mock_enthusiasm": (
        "Mock enthusiasm detected. Address the underlying frustration: "
        "'You seem frustrated. What's not working?'"
    ),
    "dismissive": (
        "Short dismissive response. Dig deeper: 'Can you tell me more "
        "about that?'"
    ),
    "contradiction": (
        "Mixed signals in that answer. Check what they actually mean: "
        "'What's the real answer there?'"
    ),
}

_DEFAULT_SARCASM_PROMPT = (
    "Sarcasm detected - may indicate resistance. Explore it: "
    "'What's really going on here?'"
)

_FALLBACK = [
    "Continue with active listening",
    "Ask 'What else?' to explore deeper",
    "Reflect back what you are hearing",
]


class SuggestionBuilder:
    """Combines the contextual engine with signal-specific prompts."""

    def __init__(self, engine, gemini_analyzer=None, gemini_every: int = 5):
        self.engine = engine
        self.gemini_analyzer = gemini_analyzer
        self.gemini_every = gemini_every

    async def build(
        self,
        chunk,
        inferences,
        grow_phase,
        history: Sequence[Any],
        sarcasm: Optional[Dict[str, Any]] = None,
        digression: Optional[Dict[str, Any]] = None,
        turn_index: int = 0,
    ) -> List[str]:
        try:
            suggestions = list(self.engine.generate_suggestions(
                chunk=chunk,
                inferences=inferences,
                grow_phase=grow_phase,
                conversation_history=list(history),
            ))
        except Exception as exc:
            logger.error("Contextual suggestions failed: %s", exc, exc_info=True)
            suggestions = list(_FALLBACK)

        if digression and digression.get("is_digression") and chunk.speaker == "coachee":
            suggestions.insert(0, (
                "Conversation drifted off topic. Acknowledge briefly, then "
                "steer back: 'Let's park that - back to what you said about...'"
            ))

        if sarcasm and sarcasm.get("is_sarcastic"):
            suggestions.insert(0, self._sarcasm_prompt(chunk.speaker, sarcasm))

        ai_suggestion = await self._gemini_suggestion(
            chunk, grow_phase, history, sarcasm, turn_index
        )
        if ai_suggestion:
            suggestions.insert(0, ai_suggestion)

        return suggestions[:_MAX_SUGGESTIONS]

    @staticmethod
    def _sarcasm_prompt(speaker: str, sarcasm: Dict[str, Any]) -> str:
        if speaker == "coach":
            return "Your tone may read as sarcastic. Stay authentic and supportive."
        return _SARCASM_PROMPTS.get(sarcasm.get("type", ""), _DEFAULT_SARCASM_PROMPT)

    async def _gemini_suggestion(
        self, chunk, grow_phase, history, sarcasm, turn_index: int
    ) -> Optional[str]:
        """Occasional LLM suggestion; never blocks the pipeline on failure."""
        if not self.gemini_analyzer or not getattr(self.gemini_analyzer, "model", None):
            return None
        if chunk.speaker != "coach" or turn_index % self.gemini_every != 0:
            return None

        try:
            context = "\n".join(
                f"{c.speaker}: {c.transcript}" for c in list(history)[-5:]
            )
            sarcasm_note = (
                "Sarcasm detected in the coachee's response - may indicate resistance."
                if sarcasm and sarcasm.get("is_sarcastic") else ""
            )
            prompt = (
                "You are an expert coaching advisor. Based on this conversation, "
                "give ONE brief, actionable suggestion for the coach.\n\n"
                f"Recent conversation:\n{context}\n\n"
                f"Current GROW phase: {grow_phase.phase}\n{sarcasm_note}\n\n"
                "Provide ONE specific, actionable suggestion (max 15 words)."
            )
            response = await self.gemini_analyzer.model.generate_content_async(prompt)
            text = (response.text or "").strip().replace("\n", " ")[:150]
            return f"AI: {text}" if text else None
        except Exception as exc:
            logger.warning("Gemini suggestion failed: %s", exc)
            return None
