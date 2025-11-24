import streamlit as st
import joblib
import os
from src.predict import predict, load_artifacts

# Set page config
st.set_page_config(page_title="Smart Support Agent", page_icon="🤖")

st.title("🤖 Smart Support Agent")
st.write("Enter a customer support ticket to classify its category.")

# Load model
@st.cache_resource
def get_model():
    try:
        return load_artifacts("models")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, vectorizer = get_model()

# User input
user_input = st.text_area("Customer Complaint / Ticket:", height=150)

if st.button("Classify Ticket"):
    if user_input:
        if model and vectorizer:
            prediction = predict(user_input, model, vectorizer)
            st.success(f"Predicted Category: **{prediction}**")
        else:
            st.error("Model not loaded. Please train the model first.")
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.markdown("### About")
st.write("This agent uses a Logistic Regression model trained on customer support tickets to classify them into categories like Billing, Technical Issue, etc.")
