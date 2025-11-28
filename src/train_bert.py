import pandas as pd
import numpy as np
import torch
import os
import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- CONFIGURATION ---
DAGSHUB_USER = "ErrakrakiMohamed"
REPO_NAME = "Smart-Support-Agent-Multi-Class-Text-Classification"

# Setup MLflow
os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{DAGSHUB_USER}/{REPO_NAME}.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = "66xxxxxxxxxxxxxxxxxxxxx4908ceedb"

# Model Setup
MODEL_NAME = "distilbert-base-uncased"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # 1. Load Data
    print("🔄 Loading Processed Data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # 2. Prepare Datasets
    # Hugging Face needs a specific Dataset object
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    # 3. Tokenization
    print("🔠 Tokenizing data (Converting text to IDs)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], padding="max_length", truncation=True, max_length=64)

    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    # 4. Initialize Model
    num_labels = train_df['label'].nunique()
    print(f"🧠 Loading DistilBERT with {num_labels} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,              # 3 loops
        per_device_train_batch_size=16,  # Batch size
        per_device_eval_batch_size=64,
        eval_strategy="epoch",     # Check accuracy every epoch
        save_strategy="epoch",           # Save model every epoch
        learning_rate=2e-5,              # Low learning rate for fine-tuning
        load_best_model_at_end=True,     # Keep the best one
        metric_for_best_model="accuracy",
        report_to="mlflow",              # <--- Log to DagsHub
        run_name="DistilBERT_Run",       # Name in the dashboard
        logging_dir='./logs',
        no_cuda=False if torch.cuda.is_available() else True # Use GPU if possible
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    print("🚀 Starting Training (This may take a while)...")
    mlflow.set_experiment("Support-Agent-Benchmark")
    trainer.train()

    # 8. Final Test Evaluation (The "Battle" Step)
    print("\n🧪 Running evaluation on TEST set...")
    test_result = trainer.predict(test_ds)
    print("--- DistilBERT Test Results ---")
    print(test_result.metrics)

    # Log Final Test Metrics to MLflow manually (since Trainer logs val by default)
    mlflow.log_metrics({
        "test_accuracy": test_result.metrics["test_accuracy"],
        "test_f1": test_result.metrics["test_f1"]
    })

    # 9. Save Model
    print("💾 Saving model...")
    save_path = "/content/distilbert-intent-classifier"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("✅ Done!")

if __name__ == "__main__":
    main()