import joblib
import os
import re

def load_artifacts(model_dir="models"):
    """
    Loads the trained model and vectorizer.
    """
    model_path = os.path.join(model_dir, "model.joblib")
    vec_path = os.path.join(model_dir, "vectorizer.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError("Model artifacts not found. Please train the model first.")
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

def clean_text(text):
    # Duplicate cleaning logic or import from preprocessing
    # Ideally import, but for simplicity here:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def predict(text, model, vectorizer):
    """
    Predicts the intent of the given text.
    """
    cleaned_text = clean_text(text)
    vec_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vec_text)
    return prediction[0]

if __name__ == "__main__":
    # Test run
    try:
        model, vectorizer = load_artifacts()
        test_text = "I have a problem with my bill."
        print(f"Text: {test_text}")
        print(f"Prediction: {predict(test_text, model, vectorizer)}")
    except Exception as e:
        print(e)
