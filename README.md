# 🤖 Smart Support Agent: End-to-End MLOps NLP Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Hugging_Face-Transformers-yellow)
![MLOps](https://img.shields.io/badge/MLOps-DVC_%26_DagsHub-green)
![App](https://img.shields.io/badge/Frontend-Streamlit-red)

## 📌 Project Overview
The **Smart Support Agent** is an NLP-based system designed to automatically classify customer support tickets into **27 distinct intent categories** (e.g., *cancel_order*, *payment_issue*, *change_password*).

Unlike a standard data science notebook, this project is built as a **production-ready MLOps pipeline**. It features data versioning, experiment tracking, reproducible training scripts, and a deployed web application.

### 🎯 Key Features
* **Multi-Class Text Classification:** accurately routes customer queries.
* **MLOps Architecture:** Fully reproducible pipeline using **DVC** (Data Version Control).
* **Experiment Tracking:** Model metrics logged via **MLflow** & **DagsHub**.
* **Interactive Demo:** A real-time web interface built with **Streamlit**.

---

## 🏗️ Architecture & Tools

This project moves beyond "Jupyter Notebooks" by implementing a modular codebase:

| Component | Tool Used | Purpose |
| :--- | :--- | :--- |
| **Data Versioning** | **DVC** | Tracks large datasets and model files that Git cannot handle. |
| **Storage & Tracking** | **DagsHub** | Acts as the remote storage for DVC and MLflow server. |
| **Model Training** | **PyTorch / Hugging Face** | Fine-tuning the DistilBERT transformer model. |
| **Experimentation** | **MLflow** | Comparing Baseline vs. Deep Learning models. |
| **Deployment** | **Streamlit** | Serving the model in a user-friendly web app. |

---

## ⚔️ The "Battle of the Models"

To ensure the best solution, we benchmarked a traditional approach against a modern Deep Learning approach.

**The Data:** [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (27k examples).

| Model | Architecture | Preprocessing | Accuracy | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Logistic Regression | TF-IDF (Stopwords removed) | **97%** | Fast, but struggles with complex phrasing. |
| **Champion** | **DistilBERT (Fine-Tuned)** | Raw Text (Context preserved) | **99%** | **Chosen for deployment.** Human-level understanding. |

> *Note: Experiments were tracked and visualized using MLflow.*

---

## 🚀 Installation & Setup

Since large files are stored in DVC, simply cloning the repo is not enough. Follow these steps to reproduce the environment.

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/Smart-Support-Agent.git](https://github.com/ErrakrakiMohamed/Smart-Support-Agent-Multi-Class-Text-Classification.git)
cd Smart-Support-Agent

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Data & Models 

```bash
dvc pull
```

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

📂 Project Structure

Smart-Support-Agent/
├── .dvc/                 # DVC configuration
├── data/                 # Data folder (Tracked by DVC)
│   ├── raw_data.csv.dvc  # Pointer to raw data
│   └── processed/        # Cleaned train/test splits
├── models/               # Model folder (Tracked by DVC)
│   ├── baseline/         # Saved Logistic Regression model
│   └── distilbert/       # Saved DistilBERT model (250MB)
├── notebooks/            # Experimental Jupyter Notebooks
├── src/                  # Source Code
│   ├── get_data.py       # Data Ingestion script
│   ├── preprocess.py     # Cleaning & Splitting pipeline
│   ├── train_baseline.py # Training script for Logistic Regression
│   └── train_bert.py     # Training script for DistilBERT
├── app.py                # Streamlit Application
├── requirements.txt      # Python dependencies
└── README.md             # Project Documentation

👨‍💻 Usage
Open the Streamlit App.

Type a customer complaint (e.g., "I have not received my refund yet").

Click "Classify Intent".

The AI will display the predicted category (e.g., payment_issue) and its confidence score.

📜 Credits
Dataset: Bitext (Hugging Face).

Tools: DagsHub, DVC, Streamlit.

Author: Mohamed Errakraki