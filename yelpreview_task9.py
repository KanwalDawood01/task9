import os
os.environ['CURL_CA_BUNDLE'] = ""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "ModelZoo/YelpTrainerMinimal/checkpoint-315"  # Adjust if different

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir="./CentralCache")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model

@st.cache_data
def load_data():
    dataset = load_dataset("yelp_review_full", cache_dir="./CentralCache")
    train = dataset["train"].shuffle(seed=42).select(range(1000))
    test = dataset["test"].shuffle(seed=42).select(range(1000))
    return train, test

def predict_review(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
    return prediction, probs.squeeze().cpu().numpy()

# Load components
tokenizer, model = load_model_and_tokenizer()
train_data, test_data = load_data()

# UI
st.set_page_config(page_title="Yelp Review Sentiment Classifier", layout="wide")
st.title("⭐ Yelp Review Classification using BERT")
st.markdown("This app predicts **review ratings (1–5 stars)** from text using a fine-tuned BERT model.")

# Tabs for Manual Input and Dataset Samples
tab1, tab2 = st.tabs(["✍️ Manual Review", "📊 Dataset Samples"])

with tab1:
    st.subheader("Write your review below:")
    user_input = st.text_area("Enter your review text:", "This is the best restaurant I’ve ever visited!", height=150)

    if st.button("Predict Rating"):
        if user_input.strip() != "":
            pred_label, prob = predict_review(user_input, tokenizer, model)
            st.success(f"🌟 Predicted Rating: **{pred_label + 1} stars**")
            st.bar_chart(prob)

with tab2:
    st.subheader("Sample Train & Test Reviews")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧪 Train Samples")
        selected_train = st.selectbox("Select a train review", [f"{ex['text'][:100]}... (Label: {ex['label'] + 1})" for ex in train_data])
        idx = [f"{ex['text'][:100]}... (Label: {ex['label'] + 1})" for ex in train_data].index(selected_train)
        st.write("**Full Review:**", train_data[idx]["text"])
        st.write("**Ground Truth:**", train_data[idx]["label"] + 1)
        pred_label, _ = predict_review(train_data[idx]["text"], tokenizer, model)
        st.write("**Predicted Label:**", pred_label + 1)

    with col2:
        st.markdown("### 🧪 Test Samples")
        selected_test = st.selectbox("Select a test review", [f"{ex['text'][:100]}... (Label: {ex['label'] + 1})" for ex in test_data])
        idx = [f"{ex['text'][:100]}... (Label: {ex['label'] + 1})" for ex in test_data].index(selected_test)
        st.write("**Full Review:**", test_data[idx]["text"])
        st.write("**Ground Truth:**", test_data[idx]["label"] + 1)
        pred_label, _ = predict_review(test_data[idx]["text"], tokenizer, model)
        st.write("**Predicted Label:**", pred_label + 1)

st.markdown("---")
st.caption("Developed for CS-878 | Week-10 | Fine-Tuned BERT on Yelp Review Full Dataset")
