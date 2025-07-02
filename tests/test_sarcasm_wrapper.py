import unittest
from unittest.mock import MagicMock
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_wrappers import SarcasmDetectionWrapper

class TestSarcasmDetectionWrapper(unittest.TestCase):
    def setUp(self):
        # Create a fake model with predict_proba()
        class MockModel:
            def predict_proba(self, inputs):
                return [[0.2, 0.8]]  # Always 80% sarcastic
        
        # Create a mock wrapper without loading a real model
        self.wrapper = SarcasmDetectionWrapper(model_path=r"C:\Users\vedan\Desktop\coachingSystem\models\saved_models\model_lstm.pkl")
        self.wrapper.model = MockModel()
        self.wrapper.is_loaded = True
        self.wrapper.tokenizer = self._mock_tokenizer()
        self.wrapper.max_length = 128

    def _mock_tokenizer(self):
        class MockTokenizer:
            def encode_plus(self, text, **kwargs):
                return {
                    'input_ids': [[101, 2009, 2003, 1037, 2329, 102]],
                    'attention_mask': [[1, 1, 1, 1, 1, 1]]
                }
        return MockTokenizer()

    def test_string_input(self):
        is_sarcastic, confidence = self.wrapper.predict("I'm so thrilled to be stuck in traffic!")
        self.assertTrue(is_sarcastic)
        self.assertGreater(confidence, 0.5)

    def test_list_input(self):
        is_sarcastic, confidence = self.wrapper.predict(["Oh", "great", "another", "Monday"])
        self.assertTrue(is_sarcastic)
        self.assertGreater(confidence, 0.5)

    def test_multiline_input(self):
        text = ["Right now, my mornings are inconsistent—I’d rate them a 4/10.",
                "Sleep schedule varies.",
                "Mindset: check emails first thing."]
        is_sarcastic, confidence = self.wrapper.predict(text)
        self.assertTrue(is_sarcastic)
        self.assertGreater(confidence, 0.5)

    def test_empty_input(self):
        is_sarcastic, confidence = self.wrapper.predict("")
        self.assertFalse(is_sarcastic)
        self.assertEqual(confidence, 0.0)

    def test_non_string_input(self):
        is_sarcastic, confidence = self.wrapper.predict(12345)
        self.assertTrue(is_sarcastic)
        self.assertGreater(confidence, 0.5)

if __name__ == "__main__":
    unittest.main()
