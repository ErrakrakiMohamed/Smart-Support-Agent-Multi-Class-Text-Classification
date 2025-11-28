import streamlit as st
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
MODEL_PATH = "models/distilbert-intent-classifier"
LABEL_MAP_PATH = "data/processed/label_map.json"

# --- PAGE SETUP ---
st.set_page_config(page_title="Smart Support Agent", page_icon="🤖")
st.title("🤖 Smart Customer Support Agent")
st.write("Enter a customer message below to classify its intent.")

# --- LOAD RESOURCES (Cached for speed) ---
@st.cache_resource
def load_resources():
    # 1. Load the Label Mapping
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    # Invert it: {0: "cancel_order"} instead of {"cancel_order": 0}
    id2label = {v: k for k, v in label_map.items()}
    
    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model, id2label

# Show a loading spinner while the model loads
with st.spinner("Loading AI Brain..."):
    tokenizer, model, id2label = load_resources()

# --- USER INPUT ---
user_text = st.text_area("Customer Message:", "I have not received my refund yet.")

# --- PREDICTION ---
if st.button("Classify Intent"):
    if not user_text:
        st.warning("Please enter some text.")
    else:
        # 1. Tokenize
        inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        
        # 2. Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 3. Get Probabilities (Softmax)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_prob, top_idx = torch.max(probs, dim=-1)
        
        # 4. Get Label Name
        predicted_label = id2label[top_idx.item()]
        confidence = top_prob.item() * 100
        
        # --- DISPLAY RESULT ---
        st.success(f"**Predicted Intent:** {predicted_label}")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        
        # Optional: Debug info
        with st.expander("See Raw Probabilities"):
            # Convert to list for plotting
            probs_np = probs.detach().numpy()[0]
            labels = [id2label[i] for i in range(len(probs_np))]
            
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Create a DataFrame for easier sorting/plotting
            df_probs = pd.DataFrame({"Label": labels, "Probability": probs_np})
            df_probs = df_probs.sort_values(by="Probability", ascending=True)
            
            # Plot
            fig, ax = plt.subplots()
            ax.barh(df_probs["Label"], df_probs["Probability"], color="skyblue")
            ax.set_xlabel("Probability")
            ax.set_title("Intent Probabilities")
            st.pyplot(fig)