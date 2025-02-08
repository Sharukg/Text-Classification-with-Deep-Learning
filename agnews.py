import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd


# Load Tokenizer and Models
maxlen = 200
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
cnn_model = tf.keras.models.load_model("cnn_ag_news.h5")
han_model = tf.keras.models.load_model("han_ag_news.h5")

# Streamlit App Layout
st.title("AG News Classification App")
st.subheader("Classify news articles into 4 categories: World, Sports, Business, Science/Technology")

# User Input
text_input = st.text_area("Enter a news article:")
model_choice = st.selectbox("Choose a Model:", ["CNN", "HAN"])

if st.button("Classify"):
    if text_input.strip():
        sequence = tokenizer.texts_to_sequences([text_input])
        padded_sequence = pad_sequences(sequence, maxlen=maxlen)
        
        model = cnn_model if model_choice == "CNN" else han_model
        prediction = model.predict(padded_sequence)
        category = np.argmax(prediction)
        labels = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}
        
        st.write(f"### Prediction: {labels[category]}")
        st.bar_chart(prediction[0])
    else:
        st.warning("Please enter a valid text input.")

# Model Performance Visualization
st.subheader("Model Performance Comparison")
results = {
    "Model": ["CNN", "HAN"],
    "Accuracy": [0.85, 0.88],  # Example values, replace with actual
    "Inference Time (sec)": [0.02, 0.08]
}

df = sns.load_dataset("titanic")[['sex', 'age']].dropna().sample(2, replace=True)  # Dummy replacement
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

sns.barplot(x="Model", y="Accuracy", data=pd.DataFrame(results), ax=ax[0], palette="viridis")
ax[0].set_title("Model Accuracy Comparison")

sns.barplot(x="Model", y="Inference Time (sec)", data=pd.DataFrame(results), ax=ax[1], palette="magma")
ax[1].set_title("Inference Time Comparison")

st.pyplot(fig)
