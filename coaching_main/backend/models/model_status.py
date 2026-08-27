"""
Honest reporting of what each model is actually doing.

The previous engine marked a model ``"loaded"`` whenever its wrapper class
could be *constructed*. Every wrapper probed for files that were never
shipped, found none, raised nothing, and was reported as loaded - so the
API and dashboard said "all models loaded" while three of the four were
returning hardcoded fallbacks. Nothing in the logs, the API or the UI
indicated otherwise.

A model now reports one of three states, and the reason is always attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelState(str, Enum):
    """What is actually producing this model's predictions."""

    #: Trained weights loaded; predictions come from the model.
    TRAINED = "trained"
    #: Weights unusable; a documented rule-based heuristic is running instead.
    HEURISTIC = "heuristic"
    #: Neither weights nor a fallback - this signal is not being produced.
    UNAVAILABLE = "unavailable"


@dataclass
class ModelStatus:
    """Load outcome for one model, including why it turned out that way."""

    name: str
    state: ModelState
    detail: str = ""
    weights_loaded: Optional[str] = None
    artifacts_found: List[str] = field(default_factory=list)
    artifacts_missing: List[str] = field(default_factory=list)
    blocking_reason: Optional[str] = None

    @property
    def is_trained(self) -> bool:
        return self.state is ModelState.TRAINED

    @property
    def is_degraded(self) -> bool:
        return self.state is not ModelState.TRAINED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "detail": self.detail,
            "weights_loaded": self.weights_loaded,
            "artifacts_found": self.artifacts_found,
            "artifacts_missing": self.artifacts_missing,
            "blocking_reason": self.blocking_reason,
        }

    def log_line(self) -> str:
        icon = {"trained": "OK", "heuristic": "DEGRADED", "unavailable": "MISSING"}
        prefix = icon.get(self.state.value, "?")
        line = f"[{prefix}] {self.name}: {self.detail}"
        if self.blocking_reason:
            line += f" | blocked by: {self.blocking_reason}"
        return line


def trained(name: str, weights: str, detail: str = "") -> ModelStatus:
    return ModelStatus(
        name=name,
        state=ModelState.TRAINED,
        detail=detail or f"trained weights loaded from {weights}",
        weights_loaded=weights,
    )


def heuristic(
    name: str,
    blocking_reason: str,
    detail: str = "",
    found: Optional[List[str]] = None,
    missing: Optional[List[str]] = None,
) -> ModelStatus:
    return ModelStatus(
        name=name,
        state=ModelState.HEURISTIC,
        detail=detail or "running rule-based heuristic, not the trained model",
        artifacts_found=found or [],
        artifacts_missing=missing or [],
        blocking_reason=blocking_reason,
    )


def unavailable(name: str, blocking_reason: str, detail: str = "") -> ModelStatus:
    return ModelStatus(
        name=name,
        state=ModelState.UNAVAILABLE,
        detail=detail or "no prediction available",
        blocking_reason=blocking_reason,
    )


def summarize(statuses: Dict[str, ModelStatus]) -> Dict[str, Any]:
    """Aggregate view for the ``/model-status`` endpoint and the dashboard."""
    degraded = [s.name for s in statuses.values() if s.is_degraded]
    return {
        "models": {name: s.to_dict() for name, s in statuses.items()},
        "all_trained": not degraded,
        "degraded": degraded,
        "trained_count": sum(1 for s in statuses.values() if s.is_trained),
        "total_count": len(statuses),
    }
