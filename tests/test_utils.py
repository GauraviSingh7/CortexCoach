import numpy as np
from utils.image_processing import preprocess_facial_image

def test_preprocess_facial_image_output_shape():
    dummy_image = (np.ones((100, 100, 3)) * 255).astype("uint8")
    result = preprocess_facial_image(dummy_image)
    assert result.shape == (1, 48, 48, 1)
    assert result.dtype == "float32"
