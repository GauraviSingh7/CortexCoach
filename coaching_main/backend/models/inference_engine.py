"""
Model inference engine.

Loads the four model adapters, reports honestly what each one is actually
doing, and runs them concurrently over a chunk.

Two rules this module now enforces:

* A model is only reported as ``trained`` when real weights are in memory.
  The previous version marked a model ``"loaded"`` whenever its wrapper
  class could be constructed, which was always - every wrapper probed for
  filenames that were never shipped, found nothing, and raised nothing.

* Adapters return ``None`` for "no model output". This engine passes that
  ``None`` straight through instead of substituting a default, so the
  orchestrator can choose a labelled fallback. Manufacturing
  ``{"neutral": 1.0}`` here is what made every turn read as 100% neutral.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import joblib

from backend.models.model_status import ModelState, ModelStatus, summarize, unavailable
from backend.schemas.data_models import AudioChunk, ModelInferences

# Model classes must be importable before their pickles are opened so that
# pickle can resolve the class references stored inside them.
from models.emotion_recognition.inference import EmotionRecognitionModel
from models.interest_detection.inference import EngagementPredictor, InterestDetectionModel
from models.sarcasm_detection.inference import SarcasmDetectionModel
from models.vak_inference.inference import VAKInferenceModel

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when a model directory is missing entirely."""


class CustomUnpickler(pickle.Unpickler):
    """Resolves classes pickled from ``__main__`` in a training script."""

    _REDIRECTS = {
        "__main__": "models.interest_detection.inference",
        "__mp_main__": "models.interest_detection.inference",
    }

    def find_class(self, module: str, name: str):
        if module in self._REDIRECTS and name == "EngagementPredictor":
            module = self._REDIRECTS[module]
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            if name == "EngagementPredictor":
                return EngagementPredictor
            raise


def load_pickle_with_compatibility(file_path: Path):
    """Open a pickle written by a training script, tolerating stale paths."""
    # The shipped pipeline is a joblib (numpy-pickle) file, so try joblib
    # first; CustomUnpickler is the fallback for plain pickles written by a
    # training script under __main__.
    try:
        return joblib.load(file_path)
    except Exception as joblib_error:
        logger.debug("joblib could not read %s (%s); trying custom unpickler",
                     file_path, joblib_error)
        with open(file_path, "rb") as handle:
            return CustomUnpickler(handle).load()


class ModelInferenceEngine:
    """Loads the model adapters and runs them over audio chunks."""

    def __init__(self, models_base_path: str = "./models"):
        self.models_base_path = Path(models_base_path)
        self.models: Dict[str, Any] = {}
        self.statuses: Dict[str, ModelStatus] = {}
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")

        self._register_legacy_classes()
        self._load_all_models()
        self._log_status()

    # -- loading -----------------------------------------------------------

    @staticmethod
    def _register_legacy_classes() -> None:
        """Expose model classes under ``__main__`` for legacy pickles."""
        import types

        for module_name in ("__main__", "__mp_main__"):
            module = sys.modules.get(module_name)
            if module is None:
                module = types.ModuleType(module_name)
                sys.modules[module_name] = module
            module.EngagementPredictor = EngagementPredictor
            module.InterestDetectionModel = InterestDetectionModel

    def _load_all_models(self) -> None:
        loaders: Dict[str, Callable[[], Any]] = {
            "emotion_recognition": self._load_emotion_model,
            "interest_detection": self._load_interest_model,
            "sarcasm_detection": self._load_sarcasm_model,
            "vak_inference": self._load_vak_model,
        }

        for name, loader in loaders.items():
            try:
                instance = loader()
                self.models[name] = instance
                status = getattr(instance, "status", None)
                self.statuses[name] = status or unavailable(
                    name, "adapter did not report a status"
                )
            except Exception as exc:
                logger.error("Failed to load %s: %s", name, exc, exc_info=True)
                self.models[name] = None
                self.statuses[name] = unavailable(name, str(exc))

    def _model_dir(self, name: str) -> Path:
        path = self.models_base_path / name
        if not path.exists():
            raise ModelLoadError(f"model directory not found: {path}")
        return path

    def _load_emotion_model(self) -> EmotionRecognitionModel:
        return EmotionRecognitionModel(self._model_dir("emotion_recognition"))

    def _load_sarcasm_model(self) -> SarcasmDetectionModel:
        return SarcasmDetectionModel(self._model_dir("sarcasm_detection"))

    def _load_vak_model(self) -> VAKInferenceModel:
        return VAKInferenceModel(self._model_dir("vak_inference"))

    def _load_interest_model(self):
        path = self._model_dir("interest_detection")
        pipeline_file = path / "engagement_pipeline.pkl"

        if pipeline_file.exists():
            try:
                loaded = load_pickle_with_compatibility(pipeline_file)
                if isinstance(loaded, (InterestDetectionModel, EngagementPredictor)):
                    return loaded
                # A bare sklearn pipeline - wrap it so .predict() works.
                wrapper = InterestDetectionModel(path)
                wrapper.text_model = loaded
                return wrapper
            except Exception as exc:
                logger.error("Interest pipeline failed to load: %s", exc)

        return InterestDetectionModel(path)

    def _log_status(self) -> None:
        """Make degraded models impossible to miss in the logs."""
        for status in self.statuses.values():
            if status.state is ModelState.TRAINED:
                logger.info("%s", status.log_line())
            else:
                logger.warning("%s", status.log_line())

        degraded = [s.name for s in self.statuses.values() if s.is_degraded]
        if degraded:
            logger.warning(
                "%d of %d models are NOT using trained weights: %s. "
                "Predictions for these come from rule-based heuristics.",
                len(degraded), len(self.statuses), ", ".join(degraded),
            )

    # -- inference ---------------------------------------------------------

    async def process_chunk(self, chunk: AudioChunk) -> ModelInferences:
        """Run every available model over one chunk.

        Fields are ``None`` when no trained model produced a value; the
        orchestrator decides on the labelled fallback.
        """
        tasks = {
            "emotion": self._run(
                self.models.get("emotion_recognition"),
                lambda m: m.predict(chunk.transcript, chunk.audio_data),
            ),
            "interest": self._run(
                self.models.get("interest_detection"),
                lambda m: m.predict(chunk.transcript, chunk.audio_data),
            ),
            "sarcasm": self._run(
                self.models.get("sarcasm_detection"),
                lambda m: m.predict(chunk.transcript),
            ),
            "vak": self._run(
                self.models.get("vak_inference"),
                lambda m: m.predict(chunk.transcript),
            ),
        }

        completed = await asyncio.gather(*tasks.values(), return_exceptions=True)
        results: Dict[str, Any] = {}
        for name, outcome in zip(tasks, completed):
            if isinstance(outcome, Exception):
                logger.error("Inference error for %s: %s", name, outcome)
                results[name] = None
            else:
                results[name] = outcome

        interest = results.get("interest")
        return ModelInferences(
            emotion=results.get("emotion"),
            interest_level=float(interest) if interest is not None else None,
            sarcasm_score=results.get("sarcasm"),
            vak_style=results.get("vak"),
            digression_score=None,
            emotion_source="model" if results.get("emotion") else None,
            sarcasm_source="model" if results.get("sarcasm") is not None else None,
            vak_source="model" if results.get("vak") else None,
        )

    async def _run(self, model: Optional[Any], call: Callable[[Any], Any]):
        """Run one model call on the thread pool; ``None`` if unavailable."""
        if model is None:
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, call, model)

    # -- reporting ---------------------------------------------------------

    def get_model_status(self) -> Dict[str, Any]:
        """Full status payload for ``/model-status`` and the dashboard."""
        return summarize(self.statuses)

    def status_of(self, name: str) -> Optional[ModelStatus]:
        return self.statuses.get(name)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
