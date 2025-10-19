
import streamlit as st
import os
from audio_utils import extract_features
from gemini_engine import classify_raga

st.title("🎶 Raga Classifier AI")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    file_path = os.path.join("data/samples", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path)
    st.write("Extracting features...")
    features = extract_features(file_path)
    st.json(features)

    st.write("Classifying raga using Gemini...")
    result = classify_raga(features)
    st.success(result)
