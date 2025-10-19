import streamlit as st
import os
from audio_utils import extract_features
from gemini_engine import classify_raga

st.set_page_config(page_title="Raga Classifier AI", layout="centered")

# Sidebar with app description
st.sidebar.title("About")
st.sidebar.info("""
🎶 RagaClassifierAI uses audio features and Gemini AI to classify Indian classical ragas.
Upload a clip, extract features, and get intelligent predictions.
""")

# Optional manual raga override
manual_raga = st.sidebar.selectbox("Select Raga (optional override)", ["None", "Yaman", "Bhairav", "Durga"])

st.title("🎵 Raga Classifier AI")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    file_path = os.path.join("data/samples", uploaded_file.name)
    os.makedirs("data/samples", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path)
    st.write("Extracting features...")
    features = extract_features(file_path)
    st.json(features)

    st.write("Classifying raga using Gemini...")
    result = classify_raga(features)

    # Try to extract confidence if present
    confidence = None
    if "confidence:" in result.lower():
        import re
        match = re.search(r"confidence[:\\s]+(\\d+%?)", result, re.IGNORECASE)
        if match:
            confidence = match.group(1)

    # Display result and confidence
    st.success(result)
    if confidence:
        st.metric("Confidence", confidence)

    # Show manual override if selected
    if manual_raga != "None":
        st.warning(f"Manual override selected: {manual_raga}")
