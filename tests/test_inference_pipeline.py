from models.inference_pipeline import MultimodalInferencePipeline
import numpy as np

def test_pipeline_with_text_and_image():
    pipeline = MultimodalInferencePipeline({
        'facial_emotion': 'models/saved_models/emotion_model.h5',
        'sarcasm': 'models/saved_models/best_sarcasm_model.h5',
        'vak': 'models/saved_models/label_encoder.pkl'
    })

    dummy_image = (np.random.rand(48, 48, 3) * 255).astype("uint8")
    result = pipeline.process_multimodal_input(
        text="Wow, what a great idea.",
        image=dummy_image
    )

    assert isinstance(result, dict)
    assert "sarcasm_detected" in result
    assert "facial_emotion" in result
    assert "vak_type" in result
