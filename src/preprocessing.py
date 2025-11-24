import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

def clean_text(text):
    """
    Basic text cleaning: lowercase, remove special chars.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def preprocess_data(input_path="data/raw/tickets.csv", output_dir="data/processed"):
    """
    Loads raw data, cleans text, and splits into train/test.
    """
    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    # Identify text and label columns
    # Based on bitext dataset, columns might be 'instruction', 'intent', etc.
    # We'll try to auto-detect or default to likely names.
    text_col = 'instruction' if 'instruction' in df.columns else 'text'
    label_col = 'intent' if 'intent' in df.columns else 'label'
    
    if text_col not in df.columns or label_col not in df.columns:
        print(f"Columns {text_col} or {label_col} not found. Available: {df.columns.tolist()}")
        # Fallback for demo if columns don't match
        if len(df.columns) >= 2:
            text_col = df.columns[0]
            label_col = df.columns[1]
            print(f"Using {text_col} as text and {label_col} as label.")
    
    print("Cleaning text...")
    df['clean_text'] = df[text_col].apply(clean_text)
    
    print("Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)
    
    print(f"Data processed and saved to {output_dir}")

if __name__ == "__main__":
    preprocess_data()
