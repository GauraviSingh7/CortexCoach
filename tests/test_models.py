import numpy as np
from models.model_wrappers import FacialEmotionWrapper, SarcasmDetectionWrapper, VAKLearningStyleWrapper

def test_facial_emotion_prediction():
    model = FacialEmotionWrapper("models/saved_models/emotion_model.h5")
    dummy_image = (np.random.rand(48, 48) * 255).astype("uint8")
    result = model.predict(dummy_image)
    assert isinstance(result, dict)
    assert all(label in result for label in model.emotion_labels)

def test_sarcasm_model_load_and_predict():
    model = SarcasmDetectionWrapper("models/saved_models/best_sarcasm_model.h5")
    is_sarcastic, confidence = model.predict("Yeah, I totally love debugging at 3am.")
    assert isinstance(is_sarcastic, bool)
    assert 0.0 <= confidence <= 1.0

def test_vak_model_load_and_predict():
    model = VAKLearningStyleWrapper("models/saved_models/label_encoder.pkl")
    vak, conf = model.predict("I prefer reading from textbooks.")
    assert vak in ["visual", "auditory", "reading", "kinesthetic"]
    assert 0.0 <= conf <= 1.0
