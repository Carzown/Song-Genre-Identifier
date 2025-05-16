import streamlit as st
import numpy as np
import librosa
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from io import BytesIO
import soundfile as sf

st.set_page_config(page_title="🎵 Song Genre Identifier", layout="centered")

# --- 1. Load Model ---
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --- 2. Feature Extraction ---
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def preprocess_audio(file):
    y, sr = librosa.load(file, duration=20)
    return extract_features(y, sr).reshape(1, -1)

# --- 3. UI ---
st.title("🎶 Song Genre Identifier (GTZAN)")
st.markdown("Upload a music clip (max 20 sec). The model will try to identify its genre.")

uploaded_file = st.file_uploader("Upload your audio file (.wav)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Convert uploaded file to WAV if needed
    try:
        # Convert to WAV buffer using librosa
        y, sr = librosa.load(uploaded_file, duration=20)
        buf = BytesIO()
        sf.write(buf, y, sr, format='WAV')
        buf.seek(0)

        # Predict
        features = extract_features(y, sr).reshape(1, -1)
        prediction = model.predict(features)[0]
        st.success(f"🎧 Predicted Genre: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Error processing audio: {e}")
