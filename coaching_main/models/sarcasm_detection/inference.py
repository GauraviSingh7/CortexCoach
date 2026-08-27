"""
Sarcasm detection model adapter.

Trained-model adapter only. :meth:`predict` returns ``None`` when no usable
model is loaded; the orchestrator then falls back to the documented
heuristic in ``backend.analysis.sarcasm`` and labels the source.

The previous wrapper substituted ``ml_score = 0.5`` whenever the classifier
was missing and blended it at 60% weight, so every utterance in the session
scored a flat 0.30 - a constant that never crossed any threshold and made
the detector look like it was running when it was not.

Shipped artifact
----------------
``model_lstm.pkl`` is a pickled **Keras** functional model::

    Embedding(30000, 128) -> Bidirectional LSTM -> GlobalMaxPool -> Dense -> sigmoid
    input shape (None, 20)   output shape (None, 1)

It is loadable, but not usable: the Keras ``Tokenizer`` / ``word_index``
from training was never shipped, so there is no way to map text onto the
30k-token vocabulary the embedding expects. Feeding it arbitrary integer
ids would produce confident nonsense.

To finish this integration, drop the training tokenizer into this directory
as ``tokenizer.pkl`` (a pickled ``keras.preprocessing.text.Tokenizer``) or
``word_index.json``; this adapter picks either up automatically and
switches itself to ``trained``. ``tensorflow`` must also be installed - it
is listed as an optional extra in ``requirements.txt``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from backend.models.model_status import ModelStatus, heuristic, trained

logger = logging.getLogger(__name__)

_MODEL_FILE = "model_lstm.pkl"
_TOKENIZER_PICKLE = "tokenizer.pkl"
_WORD_INDEX_JSON = "word_index.json"

#: Sequence length the shipped network was trained with.
_SEQUENCE_LENGTH = 20


class SarcasmDetectionModel:
    """Adapter over the Keras sarcasm LSTM, when it is usable."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self.word_index: Optional[Dict[str, int]] = None
        self.status: ModelStatus = self._load()

    # -- loading -----------------------------------------------------------

    def _load(self) -> ModelStatus:
        found = sorted(p.name for p in self.model_path.glob("*") if p.is_file())
        weights = self.model_path / _MODEL_FILE

        if not weights.exists():
            return heuristic(
                "sarcasm_detection",
                blocking_reason=f"{_MODEL_FILE} not found",
                found=found,
                missing=[_MODEL_FILE],
            )

        self.word_index = self._load_word_index()
        if self.word_index is None:
            return heuristic(
                "sarcasm_detection",
                blocking_reason=(
                    f"{_MODEL_FILE} is a Keras LSTM over a 30k-token embedding, but "
                    "the training tokenizer was never shipped, so text cannot be "
                    f"mapped to its vocabulary. Add {_TOKENIZER_PICKLE} or "
                    f"{_WORD_INDEX_JSON} to this directory to enable it."
                ),
                detail="using rule-based heuristic (backend.analysis.sarcasm)",
                found=found,
                missing=[_TOKENIZER_PICKLE, _WORD_INDEX_JSON],
            )

        try:
            import pickle

            with open(weights, "rb") as handle:
                self.model = pickle.load(handle)
            return trained("sarcasm_detection", _MODEL_FILE)
        except Exception as exc:
            logger.error("Sarcasm model failed to load: %s", exc)
            self.model = None
            return heuristic(
                "sarcasm_detection",
                blocking_reason=f"{_MODEL_FILE} present but failed to load: {exc}",
                detail="using rule-based heuristic (backend.analysis.sarcasm)",
                found=found,
            )

    def _load_word_index(self) -> Optional[Dict[str, int]]:
        pickle_path = self.model_path / _TOKENIZER_PICKLE
        if pickle_path.exists():
            try:
                import pickle

                with open(pickle_path, "rb") as handle:
                    tokenizer = pickle.load(handle)
                index = getattr(tokenizer, "word_index", None)
                if index:
                    return dict(index)
            except Exception as exc:
                logger.warning("Sarcasm tokenizer unreadable: %s", exc)

        json_path = self.model_path / _WORD_INDEX_JSON
        if json_path.exists():
            try:
                return json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Sarcasm word index unreadable: %s", exc)
        return None

    # -- inference ---------------------------------------------------------

    def predict(self, text: str, context=None) -> Optional[float]:
        """Sarcasm probability, or ``None`` when no trained model is loaded."""
        if self.model is None or not self.word_index or not text:
            return None
        try:
            import numpy as np

            sequence = self._to_sequence(text)
            probability = self.model.predict(
                np.array([sequence]), verbose=0
            )[0][0]
            return float(min(max(probability, 0.0), 1.0))
        except Exception as exc:
            logger.error("Sarcasm inference failed: %s", exc)
            return None

    def _to_sequence(self, text: str):
        """Map text to padded token ids exactly as Keras' Tokenizer would."""
        import re

        tokens = re.findall(r"[a-z0-9']+", text.lower())
        ids = [
            self.word_index[t] for t in tokens
            if t in self.word_index and self.word_index[t] < 30000
        ]
        ids = ids[:_SEQUENCE_LENGTH]
        # Keras pads at the front by default.
        return [0] * (_SEQUENCE_LENGTH - len(ids)) + ids

    def get_model_info(self) -> Dict[str, object]:
        return self.status.to_dict()
