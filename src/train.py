import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os

def train_model(data_dir="data/processed", model_dir="models"):
    """
    Trains a Logistic Regression model on the processed data.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Processed data not found. Please run preprocessing first.")
        return

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Handle missing values if any
    train_df['clean_text'] = train_df['clean_text'].fillna("")
    test_df['clean_text'] = test_df['clean_text'].fillna("")

    # Assuming 'clean_text' and 'intent' columns from preprocessing
    # If columns are different, we might need to adjust.
    # The preprocessing script tries to standardize, but let's check.
    X_train = train_df['clean_text']
    y_train = train_df['intent'] # Preprocessing script should preserve this column or we need to ensure it
    X_test = test_df['clean_text']
    y_test = test_df['intent']

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    print("Saving model and vectorizer...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.joblib"))
    print(f"Model saved to {model_dir}")

if __name__ == "__main__":
    train_model()
