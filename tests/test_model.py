import pytest
import os
import joblib
from src.predict import load_artifacts, predict

# Mock data for testing
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not trained yet")
def test_model_loading():
    model, vectorizer = load_artifacts(MODEL_DIR)
    assert model is not None
    assert vectorizer is not None

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not trained yet")
def test_prediction():
    model, vectorizer = load_artifacts(MODEL_DIR)
    text = "My internet is not working"
    prediction = predict(text, model, vectorizer)
    assert isinstance(prediction, str)
    # Check if prediction is one of the expected categories if known, 
    # but for now just checking it returns a string is enough.
