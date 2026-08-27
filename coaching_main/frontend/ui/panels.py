"""
Dashboard panels that surface data quality.

The key addition is :func:`render_model_status`. The old dashboard had no
model diagnostics at all, so a session where three of four models silently
fell back to heuristics looked identical to one where every trained model
was running.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

_STATE_BADGE = {
    "trained": ("🟢", "Trained model"),
    "heuristic": ("🟡", "Rule-based heuristic"),
    "unavailable": ("🔴", "Not available"),
}

_SOURCE_NOTE = {
    "model": "from trained model",
    "heuristic": "rule-based estimate",
    "unavailable": "no signal available",
}


def render_model_status(status: Optional[Dict[str, Any]]) -> None:
    """Show which models are genuinely running and which are degraded."""
    st.subheader("🔬 Model Status")

    if not status:
        st.warning("Model status unavailable - is the backend running?")
        return

    trained = status.get("trained_count", 0)
    total = status.get("total_count", 0)
    degraded = status.get("degraded", [])

    if degraded:
        st.warning(
            f"**{trained}/{total} models are using trained weights.** "
            f"Degraded: {', '.join(degraded)}. The metrics below are computed "
            "by documented rule-based heuristics, not by the trained models."
        )
    else:
        st.success(f"All {total} models are running trained weights.")

    # Bordered containers rather than expanders: this panel is rendered
    # both standalone and inside the report's expander, and Streamlit
    # refuses to nest an expander within an expander.
    for name, model in (status.get("models") or {}).items():
        icon, label = _STATE_BADGE.get(model.get("state", ""), ("⚪", "Unknown"))
        with st.container(border=True):
            st.markdown(f"**{icon} {name}** — {label}")
            st.caption(model.get("detail") or "-")
            if model.get("weights_loaded"):
                st.caption(f"Weights: `{model['weights_loaded']}`")
            if model.get("blocking_reason"):
                st.caption(f"**Why it is not running:** {model['blocking_reason']}")
            if model.get("artifacts_missing"):
                st.caption("Missing: " + ", ".join(model["artifacts_missing"]))


def render_provenance(sources: Dict[str, str]) -> None:
    """Small legend explaining where each reported signal came from."""
    if not sources:
        return
    st.caption(
        "Signal provenance — "
        + " · ".join(
            f"**{key}**: {_SOURCE_NOTE.get(value, value)}"
            for key, value in sorted(sources.items())
        )
    )


def render_grow_coverage(coverage: Dict[str, Any]) -> None:
    """Explain how much of the session was attributable to a GROW phase."""
    if not coverage:
        return

    classified = coverage.get("classified_turns", 0)
    total = coverage.get("total_turns", 0)
    missing = coverage.get("phases_missing") or []

    left, right = st.columns(2)
    with left:
        st.metric(
            "Phase coverage",
            f"{coverage.get('coverage_pct', 0):.0f}%",
            help=f"{classified} of {total} turns were attributed to a GROW phase.",
        )
    with right:
        st.metric(
            "Phases reached",
            f"{4 - len(missing)}/4",
            help="Missing: " + (", ".join(missing) if missing else "none"),
        )

    if missing:
        st.info("Phases not reached in this session: " + ", ".join(missing))


def format_metric(value: Any, digits: int = 2) -> str:
    """Render a metric, or say plainly that it is unavailable."""
    if isinstance(value, (int, float)) and value > 0:
        return f"{value:.{digits}f}"
    return "Not Available"
