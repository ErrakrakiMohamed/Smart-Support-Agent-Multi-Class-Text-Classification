import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
INPUT_FILE = "data/raw_data.csv"
OUTPUT_DIR = "data/processed"
RANDOM_SEED = 42

def clean_tag(match):
    """
    Helper function: Takes a regex match like "{{Order Number}}"
    and converts it to "token_order_number"
    """
    # Get the text inside the brackets (e.g., "Order Number")
    content = match.group(1)
    # Replace spaces with underscores and lowercase it
    clean_content = content.replace(' ', '_').lower()
    return f"token_{clean_content}"

def clean_text_common(text):
    """
    Performs 'Canonical' cleaning suitable for BOTH models.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Handle Dataset-Specific Artifacts (The Fixed Way)
    # We find {{...}} patterns and run the 'clean_tag' function on them specifically
    text = re.sub(r'\{\{(.*?)\}\}', clean_tag, text)
    
    # 3. Remove special characters (keep letters, numbers, spaces, underscores, and basic punctuation)
    # Note: We do NOT replace spaces with underscores globally anymore!
    text = re.sub(r'[^a-z0-9\s_.,?!]', '', text)
    
    # 4. Normalize Whitespace (turn double spaces into single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    print("🧹 Starting Canonical Preprocessing (v2 - Fixed)...")
    
    # 1. Load Raw Data
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Cannot find {INPUT_FILE}. Did you run get_data.py?")
        
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Apply Common Cleaning
    print("   - Cleaning text artifacts...")
    df['clean_text'] = df['text'].apply(clean_text_common)
    
    # 3. Label Encoding
    print("   - Encoding labels...")
    label_mapping = {label: idx for idx, label in enumerate(df['category'].unique())}
    df['label'] = df['category'].map(label_mapping)
    
    # Save mapping
    import json
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_mapping, f)
        
    # 4. Stratified Split
    print("   - Splitting data...")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['category'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df['category'])
    
    # 5. Save
    print(f"   - Saving to {OUTPUT_DIR}...")
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    
    print("✅ Preprocessing Done! Data is now properly formatted (spaces preserved).")
    
    # Quick sanity check print
    print("\n--- Sample Output Check ---")
    print(train_df[['text', 'clean_text']].head(1))

if __name__ == "__main__":
    main()