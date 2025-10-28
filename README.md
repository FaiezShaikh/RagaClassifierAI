# RagaClassifierAI 🎶

An AI-powered Streamlit app that classifies Indian classical ragas using audio features and the Gemini API.

## Features

- 🎵 Upload audio clips (WAV/MP3)
- 🔍 Extract musical features (pitch, tempo, MFCCs, etc.)
- 🤖 Classify ragas using Gemini (Google's multimodal LLM)
- 🧠 Fine-tune with labeled examples for improved accuracy
- 📊 Visualize audio features and classification confidence

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/RagaClassifierAI.git
   cd RagaClassifierAI

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt


3. **Run the app:**
   ```bash
   streamlit run app.py


## Requirements

Python 3.8+
Streamlit
Librosa
NumPy, Pandas, Matplotlib
Gemini API access (via google.generativeai)

## Environment Variables
Create a .env file or set the following environment variable:
ShellGEMINI_API_KEY=your_google_generative_ai_keyShow more lines


## 📁 Project Structure
RagaClassifierAI/
├── app.py
├── utils/
│   ├── audio_processing.py
│   └── gemini_classifier.py
├── data/
│   └── sample_ragas/
├── requirements.txt
└── README.md

