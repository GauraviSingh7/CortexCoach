import cv2
import numpy as np

def preprocess_facial_image(image: np.ndarray, target_size: tuple = (48, 48)) -> np.ndarray:
    """
    Preprocess an image for facial emotion detection.

    Args:
        image (np.ndarray): Input image (BGR or grayscale).
        target_size (tuple): Size expected by the model, default (48, 48).

    Returns:
        np.ndarray: Preprocessed image with shape (1, 48, 48, 1)
    """
    if image is None:
        raise ValueError("Image is None")

    if not isinstance(image, np.ndarray):
        raise TypeError("Expected image as a NumPy array")

    # Convert to grayscale if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image
    try:
        image = cv2.resize(image, target_size)
    except Exception as e:
        raise ValueError(f"Failed to resize image: {e}")

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    # Reshape to match model input: (1, 48, 48, 1)
    image = np.expand_dims(image, axis=(0, -1))

    return image
