"""
Emotion recognition model adapter.

This wrapper is a *trained-model adapter only*. When it cannot load real
weights it says so and returns ``None`` from :meth:`predict` - it never
invents a distribution. The caller (the orchestrator) is responsible for
deciding what to do with "no model output", and for labelling whatever it
substitutes. Silently returning ``{"neutral": 1.0}`` from here is what made
every turn in the previous build read as 100% neutral.

Shipped artifact
----------------
``model_weight.pth`` is a state dict for a graph convolution network::

    conv1.lin.weight  (128, 40)     conv2.lin.weight  (128, 128)
    fc1.weight        (128, 128)    fc2.weight        (6, 128)

That is a 40-feature **audio** model with 6 output classes. Using it needs
three things this repository does not have:

* ``torch_geometric`` (the ``conv*.lin`` layout is a PyG convolution) - not
  installed and not in ``requirements.txt``;
* the graph construction used at training time, which is not documented;
* a real 40-dimensional audio feature extractor. The previous code had a
  placeholder that returned ``np.random.rand(40)``, so even the audio path
  would have been scoring pure noise.

It also cannot score text at all, which is the only input the file-based
pipeline has. Until those are supplied the adapter reports ``heuristic``
and the orchestrator uses the documented lexicon in
``backend.analysis.emotion``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from backend.models.model_status import ModelStatus, heuristic, trained

logger = logging.getLogger(__name__)

#: Filenames that would let us score text with a real classifier.
_TEXT_CLASSIFIER_FILES = ("text_emotion_model.pkl", "emotion_classifier.pkl")
_HF_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

#: The audio artifact that is present but unusable.
_AUDIO_WEIGHTS = "model_weight.pth"


class EmotionRecognitionModel:
    """Adapter over whatever emotion artifacts are actually present."""

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.text_model = None
        self.vectorizer = None
        self.labels: List[str] = []
        self.status: ModelStatus = self._load()

    # -- loading -----------------------------------------------------------

    def _load(self) -> ModelStatus:
        found = sorted(p.name for p in self.model_path.glob("*") if p.is_file())

        text_model_path = self._first_existing(_TEXT_CLASSIFIER_FILES)
        if text_model_path is not None:
            try:
                import joblib

                self.text_model = joblib.load(text_model_path)
                vectorizer = self.model_path / "vectorizer.pkl"
                if vectorizer.exists():
                    self.vectorizer = joblib.load(vectorizer)
                self.labels = self._load_labels()
                return trained("emotion_recognition", text_model_path.name)
            except Exception as exc:
                logger.error("Emotion text model failed to load: %s", exc)

        if self._first_existing(_HF_WEIGHT_FILES) is not None:
            # A transformer checkpoint would be usable; wire it here.
            return heuristic(
                "emotion_recognition",
                blocking_reason="transformer checkpoint present but no adapter implemented",
                found=found,
            )

        audio_weights = self.model_path / _AUDIO_WEIGHTS
        if audio_weights.exists():
            return heuristic(
                "emotion_recognition",
                blocking_reason=(
                    f"{_AUDIO_WEIGHTS} is a 40-feature audio graph-conv net with 6 "
                    "classes; needs torch_geometric, the training-time graph "
                    "construction, and a real audio feature extractor. It cannot "
                    "score text, which is the only input in file mode."
                ),
                detail="using text lexicon heuristic (backend.analysis.emotion)",
                found=found,
                missing=list(_TEXT_CLASSIFIER_FILES),
            )

        return heuristic(
            "emotion_recognition",
            blocking_reason="no emotion model artifacts found",
            found=found,
            missing=list(_TEXT_CLASSIFIER_FILES) + [_AUDIO_WEIGHTS],
        )

    def _first_existing(self, names) -> Optional[Path]:
        for name in names:
            candidate = self.model_path / name
            if candidate.exists():
                return candidate
        return None

    def _load_labels(self) -> List[str]:
        encoder_path = self.model_path / "label_encoder.pkl"
        if not encoder_path.exists():
            return []
        try:
            import joblib

            encoder = joblib.load(encoder_path)
            return list(getattr(encoder, "classes_", []))
        except Exception as exc:
            logger.warning("Emotion label encoder unreadable: %s", exc)
            return []

    # -- inference ---------------------------------------------------------

    def predict(
        self, text: str, audio_data: Optional[bytes] = None
    ) -> Optional[Dict[str, float]]:
        """Emotion distribution from the trained model.

        Returns ``None`` when no trained model is available, so the caller
        can fall back explicitly and label the result's provenance.
        """
        if self.text_model is None or not text:
            return None
        try:
            features = (
                self.vectorizer.transform([text]) if self.vectorizer else [text]
            )
            if not hasattr(self.text_model, "predict_proba"):
                return None
            probabilities = self.text_model.predict_proba(features)[0]
            labels = self.labels or [str(i) for i in range(len(probabilities))]
            return {
                label: float(p)
                for label, p in zip(labels, probabilities)
                if float(p) > 0.0
            }
        except Exception as exc:
            logger.error("Emotion inference failed: %s", exc)
            return None

    def get_model_info(self) -> Dict[str, str]:
        return self.status.to_dict()
