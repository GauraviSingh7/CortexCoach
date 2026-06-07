"""
Model Inference Engine - Fixed for Pickle Compatibility
======================================================
"""

import asyncio
import logging
import sys
import pickle
import joblib
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.schemas.data_models import AudioChunk, ModelInferences

# CRITICAL: Import model classes BEFORE loading pickles
# This registers them in the module namespace
from models.emotion_recognition.inference import EmotionRecognitionModel
from models.interest_detection.inference import InterestDetectionModel, EngagementPredictor
from models.sarcasm_detection.inference import SarcasmDetectionModel
from models.vak_inference.inference import VAKInferenceModel

logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Custom exception for model loading errors"""
    pass

class ModelInferenceError(Exception):
    """Custom exception for model inference errors"""
    pass


# =============================================================================
# PICKLE COMPATIBILITY FIX
# =============================================================================
class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to handle legacy class names"""
    
    def find_class(self, module, name):
        """Override find_class to redirect legacy classes"""
        
        # Map of legacy module paths to current ones
        redirects = {
            '__main__': 'models.interest_detection.inference',
            '__mp_main__': 'models.interest_detection.inference',
        }
        
        # If the module is in our redirect map, use the new path
        if module in redirects and name == 'EngagementPredictor':
            module = redirects[module]
            logger.info(f"Redirecting pickle class: {name} from old module to {module}")
        
        # Try to find the class
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError) as e:
            logger.warning(f"Could not find {module}.{name}, trying alternatives...")
            
            # Fallback: try to find it in interest_detection module
            if name == 'EngagementPredictor':
                try:
                    from models.interest_detection.inference import EngagementPredictor
                    return EngagementPredictor
                except ImportError:
                    pass
            
            # If all else fails, raise the original error
            raise e


def load_pickle_with_compatibility(file_path: Path):
    """Load pickle file with compatibility fixes"""
    try:
        with open(file_path, 'rb') as f:
            return CustomUnpickler(f).load()
    except Exception as e:
        logger.error(f"Failed to load pickle with custom unpickler: {e}")
        # Fallback to regular joblib
        return joblib.load(file_path)


# =============================================================================
# Model Inference Engine
# =============================================================================
class ModelInferenceEngine:
    """Production-ready inference engine with pickle compatibility"""
    
    def __init__(self, models_base_path: str = "./models"):
        self.models_base_path = Path(models_base_path)
        self.models = {}
        self.model_status = {}
        
        # Thread pool for concurrent inference
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")
        
        # Register legacy classes in sys.modules BEFORE loading
        self._register_legacy_classes()
        
        # Load all models
        self._load_all_models()
    
    def _register_legacy_classes(self):
        """Register legacy class names in sys.modules for pickle compatibility"""
        try:
            # Import the actual classes
            from models.interest_detection.inference import EngagementPredictor, InterestDetectionModel
            
            # Register in __main__ and __mp_main__ for multiprocessing compatibility
            if '__main__' not in sys.modules:
                import types
                main_module = types.ModuleType('__main__')
                sys.modules['__main__'] = main_module
            
            if '__mp_main__' not in sys.modules:
                import types
                mp_module = types.ModuleType('__mp_main__')
                sys.modules['__mp_main__'] = mp_module
            
            # Add the classes to these modules
            sys.modules['__main__'].EngagementPredictor = EngagementPredictor
            sys.modules['__main__'].InterestDetectionModel = InterestDetectionModel
            sys.modules['__mp_main__'].EngagementPredictor = EngagementPredictor
            sys.modules['__mp_main__'].InterestDetectionModel = InterestDetectionModel
            
            logger.info("Legacy classes registered for pickle compatibility")
            
        except Exception as e:
            logger.warning(f"Could not register legacy classes: {e}")
    
    def _load_all_models(self):
        """Load all available models with error handling"""
        model_loaders = {
            'emotion_recognition': self._load_emotion_model,
            'interest_detection': self._load_interest_model, 
            'sarcasm_detection': self._load_sarcasm_model,
            'vak_inference': self._load_vak_model
        }
        
        for model_name, loader_func in model_loaders.items():
            try:
                logger.info(f"Loading {model_name} model...")
                model_instance = loader_func()
                self.models[model_name] = model_instance
                self.model_status[model_name] = "loaded"
                logger.info(f"✅ {model_name} model loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name} model: {str(e)}", exc_info=True)
                self.models[model_name] = None
                self.model_status[model_name] = f"error: {str(e)}"
    
    def _load_emotion_model(self):
        """Load the Speech Emotion Recognition model"""
        model_path = self.models_base_path / "emotion_recognition"
        if not model_path.exists():
            raise ModelLoadError(f"Emotion model path not found: {model_path}")
        return EmotionRecognitionModel(model_path)
    
    def _load_interest_model(self):
        """Load the Interest Level Detection model"""
        model_path = self.models_base_path / "interest_detection"
        if not model_path.exists():
            raise ModelLoadError(f"Interest model path not found: {model_path}")
        
        # Try to load with compatibility fix
        pipeline_file = model_path / "engagement_pipeline.pkl"
        if pipeline_file.exists():
            try:
                # Use custom unpickler for compatibility
                loaded_model = load_pickle_with_compatibility(pipeline_file)
                logger.info(f"Loaded interest model from {pipeline_file}")
                
                # Wrap it in InterestDetectionModel if it's not already
                if isinstance(loaded_model, (InterestDetectionModel, EngagementPredictor)):
                    return loaded_model
                else:
                    # It's a raw sklearn pipeline - wrap it
                    model_wrapper = InterestDetectionModel(model_path)
                    model_wrapper.text_model = loaded_model
                    return model_wrapper
                    
            except Exception as e:
                logger.error(f"Failed to load pickle with compatibility fix: {e}")
                # Fallback to creating a new model instance
                return InterestDetectionModel(model_path)
        else:
            return InterestDetectionModel(model_path)
    
    def _load_sarcasm_model(self):
        """Load the Sarcasm Detection model"""
        model_path = self.models_base_path / "sarcasm_detection"
        if not model_path.exists():
            raise ModelLoadError(f"Sarcasm model path not found: {model_path}")
        return SarcasmDetectionModel(model_path)
    
    def _load_vak_model(self):
        """Load the VAK Learning Style Inference model"""
        model_path = self.models_base_path / "vak_inference"
        if not model_path.exists():
            raise ModelLoadError(f"VAK model path not found: {model_path}")
        return VAKInferenceModel(model_path)
    
    async def process_chunk(self, chunk: AudioChunk) -> ModelInferences:
        """Process audio chunk through all available models"""
        logger.debug(f"Processing chunk from {chunk.speaker}: {chunk.transcript[:50]}...")
        
        inference_tasks = {}
        
        if self.models.get('emotion_recognition'):
            inference_tasks['emotion'] = self._run_emotion_inference(chunk)
        
        if self.models.get('interest_detection'):
            inference_tasks['interest'] = self._run_interest_inference(chunk)
            
        if self.models.get('sarcasm_detection'):
            inference_tasks['sarcasm'] = self._run_sarcasm_inference(chunk)
            
        if self.models.get('vak_inference'):
            inference_tasks['vak'] = self._run_vak_inference(chunk)
        
        inference_tasks['digression'] = self._run_digression_inference(chunk)
        
        results = {}
        if inference_tasks:
            try:
                completed_tasks = await asyncio.gather(
                    *inference_tasks.values(),
                    return_exceptions=True
                )
                
                for task_name, result in zip(inference_tasks.keys(), completed_tasks):
                    if isinstance(result, Exception):
                        logger.error(f"Inference error for {task_name}: {result}")
                        results[task_name] = self._get_fallback_result(task_name)
                    else:
                        results[task_name] = result
                        
            except Exception as e:
                logger.error(f"Batch inference error: {e}")
                for task_name in inference_tasks.keys():
                    results[task_name] = self._get_fallback_result(task_name)
        
        return ModelInferences(
            emotion=results.get('emotion', {"neutral": 1.0}),
            interest_level=results.get('interest', 0.5),
            sarcasm_score=results.get('sarcasm', 0.0),
            vak_style=results.get('vak', {"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34}),
            digression_score=results.get('digression', 0.0)
        )
    
    async def _run_emotion_inference(self, chunk: AudioChunk) -> Dict[str, float]:
        """Run emotion recognition inference"""
        model = self.models['emotion_recognition']
        if not model:
            return {"neutral": 1.0}
            
        loop = asyncio.get_event_loop()
        try:
            emotion_probs = await loop.run_in_executor(
                self.executor, 
                model.predict, 
                chunk.transcript, 
                chunk.audio_data
            )
            return emotion_probs
        except Exception as e:
            logger.error(f"Emotion inference error: {e}")
            raise ModelInferenceError(f"Emotion model inference failed: {e}")
    
    async def _run_interest_inference(self, chunk: AudioChunk) -> float:
        """Run interest level detection inference"""
        model = self.models['interest_detection']
        if not model:
            return 0.5
            
        loop = asyncio.get_event_loop()
        try:
            interest_score = await loop.run_in_executor(
                self.executor,
                model.predict,
                chunk.transcript,
                chunk.audio_data
            )
            return float(interest_score)
        except Exception as e:
            logger.error(f"Interest inference error: {e}")
            raise ModelInferenceError(f"Interest model inference failed: {e}")
    
    async def _run_sarcasm_inference(self, chunk: AudioChunk) -> float:
        """Run sarcasm detection inference"""
        model = self.models['sarcasm_detection']
        if not model:
            return 0.0
            
        loop = asyncio.get_event_loop()
        try:
            sarcasm_score = await loop.run_in_executor(
                self.executor,
                model.predict,
                chunk.transcript
            )
            return float(sarcasm_score)
        except Exception as e:
            logger.error(f"Sarcasm inference error: {e}")
            raise ModelInferenceError(f"Sarcasm model inference failed: {e}")
    
    async def _run_vak_inference(self, chunk: AudioChunk) -> Dict[str, float]:
        """Run VAK learning style inference"""
        model = self.models['vak_inference']
        if not model:
            return {"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34}
            
        loop = asyncio.get_event_loop()
        try:
            vak_scores = await loop.run_in_executor(
                self.executor,
                model.predict,
                chunk.transcript
            )
            return vak_scores
        except Exception as e:
            logger.error(f"VAK inference error: {e}")
            raise ModelInferenceError(f"VAK model inference failed: {e}")
    
    async def _run_digression_inference(self, chunk: AudioChunk) -> float:
        """Placeholder for LLM-based digression detection"""
        return 0.1
    
    def _get_fallback_result(self, task_name: str) -> Any:
        """Get fallback results when models fail"""
        fallbacks = {
            'emotion': {"neutral": 0.6, "calm": 0.3, "uncertain": 0.1},
            'interest': 0.5,
            'sarcasm': 0.0,
            'vak': {"visual": 0.33, "auditory": 0.33, "kinesthetic": 0.34},
            'digression': 0.1
        }
        return fallbacks.get(task_name, None)
    
    def get_model_status(self) -> Dict[str, str]:
        """Get the status of all models"""
        return self.model_status.copy()
    
    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)