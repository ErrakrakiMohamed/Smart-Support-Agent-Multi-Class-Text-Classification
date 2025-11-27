import pandas as pd
import joblib
import os
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- CONFIGURATION ---
# Replace with YOUR DagsHub username
DAGSHUB_USER = "ErrakrakiMohamed" 
REPO_NAME = "Smart-Support-Agent-Multi-Class-Text-Classification"

# Setup MLflow Tracking (The Dashboard)
os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{DAGSHUB_USER}/{REPO_NAME}.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
# Tip: Ideally use an env variable, but for this specific project, pasting the token is okay
os.environ["MLFLOW_TRACKING_PASSWORD"] = "66fe7fdc90602d5e47c0db5bc83acf24908ceedb" 

def load_data():
    print("🔄 Loading Data...")
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    # Drop rows where text might be missing (safety check)
    return train.dropna(subset=['clean_text']), test.dropna(subset=['clean_text'])

def train_baseline():
    train_df, test_df = load_data()

    # 1. Start MLflow Run
    mlflow.set_experiment("Support-Agent-Benchmark")
    
    with mlflow.start_run(run_name="Baseline_LogReg"):
        print("🧠 Training Baseline Model (TF-IDF + LogReg)...")
        
        # 2. Build the Pipeline
        # - TfidfVectorizer: Converts text to numbers. We remove stopwords HERE.
        # - LogisticRegression: The classifier.
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))),
            ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
        ])
        
        # 3. Train
        pipeline.fit(train_df['clean_text'], train_df['label'])
        
        # 4. Evaluate on Test Data
        preds = pipeline.predict(test_df['clean_text'])
        
        # Calculate Metrics
        accuracy = accuracy_score(test_df['label'], preds)
        precision, recall, f1, _ = precision_recall_fscore_support(test_df['label'], preds, average='weighted')
        
        print(f"\n📊 Baseline Results:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        
        # 5. Log EVERYTHING to DagsHub
        # Parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("remove_stopwords", True)
        
        # Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Model Artifact (The file itself)
        os.makedirs("models/baseline", exist_ok=True)
        model_path = "models/baseline/model.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)
        
        print("✅ Baseline model saved and logged to DagsHub.")

if __name__ == "__main__":
    train_baseline()