import os
import pandas as pd
from datasets import load_dataset

def load_data(output_path="data/raw/tickets.csv"):
    """
    Loads the Customer Support Ticket Dataset from Hugging Face
    and saves it to a CSV file.
    """
    print("Loading dataset from Hugging Face...")
    # Using a subset or a similar dataset. 
    # 'bitext/customer-support-ticket-dataset' is a good candidate but requires auth sometimes or is large.
    # Let's use a public accessible one or a sample if that fails.
    # For this example, we'll use 'bitext/customer-support-ticket-dataset-2k-T5' which is smaller and open.
    
    try:
        dataset = load_dataset("bitext/customer-support-ticket-dataset-2k-T5", split="train")
        df = pd.DataFrame(dataset)
        
        # Rename columns to standard names if needed
        # The dataset usually has 'instruction' (text) and 'intent' (label) or similar.
        # Let's inspect columns if we could, but for now we'll just save it.
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    load_data()
