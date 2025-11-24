import pandas as pd
from datasets import load_dataset
import os

# 1. Define where to save the data
output_folder = "data"
output_file = os.path.join(output_folder, "raw_data.csv")

# 2. Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print("Downloading dataset from Hugging Face...")
# 3. Download the dataset (Bitext Customer Support)
# We only need the 'train' split for now
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

# 4. Convert to Pandas DataFrame
df = pd.DataFrame(dataset)

# 5. Select only the columns we need (Instruction = input, intent = label)
# We rename them to be clearer
df = df[['instruction', 'intent']]
df.columns = ['text', 'category']

# 6. Save to CSV
df.to_csv(output_file, index=False)

print(f"✅ Success! Data saved to {output_file}")
print(f"📊 Total examples: {len(df)}")
print(f"🏷️  Categories found: {df['category'].nunique()}")
print("Sample:")
print(df.head())