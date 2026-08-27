"""Container for everything collected during one coaching session."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.schemas.data_models import AudioChunk, RealTimeFeedback


@dataclass
class SessionState:
    """Mutable per-session record. One instance per active session."""

    session_id: str
    session_type: str = "live"
    file_path: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)

    chunks: List[AudioChunk] = field(default_factory=list)
    feedback_history: List[RealTimeFeedback] = field(default_factory=list)
    sarcasm_detections: List[Dict[str, Any]] = field(default_factory=list)
    digression_records: List[Dict[str, Any]] = field(default_factory=list)
    vak_scores: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float:
        return (datetime.now() - self.start_time).total_seconds() / 60.0

    @property
    def turn_count(self) -> int:
        return len(self.chunks)

    def recent(self, limit: int = 10) -> List[AudioChunk]:
        return self.chunks[-limit:]

    def to_report_input(
        self,
        model_status: Optional[Dict[str, Any]] = None,
        analysis_sources: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Shape the collected data for the report generators."""
        return {
            "session_id": self.session_id,
            "duration": self.duration_minutes,
            "chunks": self.chunks,
            "feedback_history": self.feedback_history,
            "vak_scores": self.vak_scores,
            "sarcasm_detections": self.sarcasm_detections,
            "digression_scores": self.digression_records,
            "model_status": model_status or {},
            "analysis_sources": analysis_sources or {},
        }
