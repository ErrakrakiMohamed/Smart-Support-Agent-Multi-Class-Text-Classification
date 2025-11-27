import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
INPUT_FILE = "data/raw_data.csv"
OUTPUT_DIR = "data/processed"
RANDOM_SEED = 42

print("Loading data...")

def clean_text_common(text):
    """
    Performs 'Canonical' cleaning suitable for BOTH models.
    We do NOT remove stopwords here because BERT needs them.
    We do NOT stem words here because BERT needs full words.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase (Standard for both, though BERT-cased exists, uncased is safer for students)
    text = text.lower()
    
    # 2. Handle Dataset-Specific Artifacts (The Bitext tags)
    # Convert "{{Order Number}}" -> "token_order_number"
    # This preserves the meaning (it's an entity) without confusing the tokenizer with brackets
    text = re.sub(r'\{\{(.*?)\}\}', r'token_\1', text)
    text = text.replace(' ', '_') # temporary join for the tag
    
    # 3. Clean up the temporary join if it affected normal text (revert underscores not in tokens)
    # Actually, a safer regex for the tags:
    text = re.sub(r'token_([a-z0-9_]+)', lambda m: m.group(0).replace(' ', '_'), text)
    
    # 4. Remove special characters but KEEP punctuation that defines sentence structure
    # BERT uses periods and commas for context. LogReg doesn't care.
    # We keep: letters, numbers, ., ?, ! and underscores (for our tokens)
    text = re.sub(r'[^a-z0-9\s_.,?!]', '', text)
    
    # 5. Normalize Whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    print("🧹 Starting Canonical Preprocessing...")
    
    # 1. Load Raw Data
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Cannot find {INPUT_FILE}. Did you run get_data.py?")
        
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Apply Common Cleaning
    print("   - Cleaning text artifacts...")
    df['clean_text'] = df['text'].apply(clean_text_common)
    
    # 3. Label Encoding (Convert "cancel_order" -> 0, 1, 2...)
    print("   - Encoding labels...")
    label_mapping = {label: idx for idx, label in enumerate(df['category'].unique())}
    df['label'] = df['category'].map(label_mapping)
    
    # Save the mapping immediately (Crucial for the App later)
    import json
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_mapping, f)
        
    # 4. Stratified Split (The "Pro" Split)
    # Train: 80%, Val: 10%, Test: 10%
    print("   - Splitting data...")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['category'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df['category'])
    
    # 5. Save the Master Datasets
    print(f"   - Saving to {OUTPUT_DIR}...")
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    
    print("✅ Preprocessing Done! We have ONE clean dataset for ALL models.")
    print(f"   Train shape: {train_df.shape}")
    print(f"   Test shape:  {test_df.shape}")

if __name__ == "__main__":
    main()
    