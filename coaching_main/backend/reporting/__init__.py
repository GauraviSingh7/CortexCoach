"""Session report assembly.

``LocalAnalyzer`` builds the full report from collected session data
without calling any external API. It is both the offline path and the
authority for every computed metric - the Gemini path narrates on top of
these numbers rather than re-deriving them.
"""

from backend.reporting.local_analyzer import LocalAnalyzer, SessionAnalysis

#: Backwards-compatible alias for the previous class name.
EnhancedLocalAnalyzer = LocalAnalyzer

__all__ = ["LocalAnalyzer", "EnhancedLocalAnalyzer", "SessionAnalysis"]
