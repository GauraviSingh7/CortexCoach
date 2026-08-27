"""
VAK learning-style model adapter.

Trained-model adapter only. :meth:`predict` returns ``None`` when no usable
model is loaded; the orchestrator then falls back to the documented
heuristic in ``backend.analysis.vak`` and labels the source.

Shipped artifacts
-----------------
This directory holds a complete BERT sequence-classification setup *except*
the weights::

    config.json              BertForSequenceClassification, 3 labels
    vocab.txt                30522-token WordPiece vocabulary
    tokenizer_config.json    BertTokenizer
    special_tokens_map.json
    label_encoder.pkl        ['Auditory', 'Kinesthetic', 'Visual']
    <MISSING>                model.safetensors / pytorch_model.bin

Neither copy of this project contains the weights file, so the model cannot
be loaded at all. Drop ``model.safetensors`` (or ``pytorch_model.bin``) in
here and this adapter switches itself to ``trained`` automatically.

Label ordering
--------------
The old code mapped ``probs[0] -> visual, probs[1] -> auditory,
probs[2] -> kinesthetic``. The label encoder is alphabetical, so the true
order is ``Auditory, Kinesthetic, Visual``. That mapping would have
silently produced plausible-looking but scrambled results the moment the
weights arrived. Labels are now read from the encoder rather than assumed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from backend.models.model_status import ModelStatus, heuristic, trained

logger = logging.getLogger(__name__)

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_CONFIG = "config.json"
_LABEL_ENCODER = "label_encoder.pkl"

#: Canonical style keys used throughout the application.
_STYLE_KEYS = {"visual": "visual", "auditory": "auditory", "kinesthetic": "kinesthetic"}


class VAKInferenceModel:
    """Adapter over the BERT VAK classifier, when its weights are present."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.labels: List[str] = []
        self.status: ModelStatus = self._load()

    # -- loading -----------------------------------------------------------

    def _load(self) -> ModelStatus:
        found = sorted(p.name for p in self.model_path.glob("*") if p.is_file())
        has_weights = any((self.model_path / f).exists() for f in _WEIGHT_FILES)
        has_config = (self.model_path / _CONFIG).exists()

        if not has_weights:
            reason = (
                "BERT config, tokenizer and label encoder are present but the "
                "weights file is missing (looked for "
                f"{' / '.join(_WEIGHT_FILES)}). The classifier cannot be "
                "instantiated without them."
                if has_config else "no VAK model artifacts found"
            )
            return heuristic(
                "vak_inference",
                blocking_reason=reason,
                detail="using keyword heuristic (backend.analysis.vak)",
                found=found,
                missing=list(_WEIGHT_FILES),
            )

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
            self.model.eval()
            self.labels = self._load_labels()
            return trained("vak_inference", "model.safetensors")
        except Exception as exc:
            logger.error("VAK model failed to load: %s", exc)
            self.model = None
            return heuristic(
                "vak_inference",
                blocking_reason=f"weights present but failed to load: {exc}",
                detail="using keyword heuristic (backend.analysis.vak)",
                found=found,
            )

    def _load_labels(self) -> List[str]:
        """Read class order from the encoder rather than assuming it."""
        encoder_path = self.model_path / _LABEL_ENCODER
        if encoder_path.exists():
            try:
                import joblib

                encoder = joblib.load(encoder_path)
                classes = list(getattr(encoder, "classes_", []))
                if classes:
                    return [str(c).lower() for c in classes]
            except Exception as exc:
                logger.warning("VAK label encoder unreadable: %s", exc)
        # Fall back to the model config, never to a positional guess.
        id2label = getattr(getattr(self.model, "config", None), "id2label", {}) or {}
        return [str(id2label[i]).lower() for i in sorted(id2label)]

    # -- inference ---------------------------------------------------------

    def predict(self, text: str) -> Optional[Dict[str, float]]:
        """VAK distribution, or ``None`` when no trained model is loaded."""
        if self.model is None or self.tokenizer is None or not text:
            return None
        try:
            import torch

            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                padding=True, max_length=512,
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0].tolist()

            scores = {"visual": 0.0, "auditory": 0.0, "kinesthetic": 0.0}
            for label, probability in zip(self.labels, probabilities):
                key = _STYLE_KEYS.get(label)
                if key:
                    scores[key] = float(probability)
            return scores if sum(scores.values()) > 0 else None
        except Exception as exc:
            logger.error("VAK inference failed: %s", exc)
            return None

    def get_model_info(self) -> Dict[str, object]:
        return self.status.to_dict()
