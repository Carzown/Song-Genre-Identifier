import streamlit as st
import numpy as np
import pickle
from dsp_utils import load_mp3, extract_features
import shutil
if not shutil.which("ffmpeg"):
    raise EnvironmentError("FFmpeg not found. Please install it and make sure it's in your PATH.")



# ---------------- Load Model ---------------- #
@st.cache_resource
def load_model():
    with open("model/genre_classifier.pkl", "rb") as f:
        return pickle.load(f)

# ---------------- Streamlit UI ---------------- #
st.title("🎧 Music Genre Classifier with DSP")

st.markdown("""
Upload an `.mp3` file. The app will apply a **low-pass filter** to the audio, extract features using **Digital Signal Processing (DSP)**, and predict the **music genre**.
""")

uploaded_file = st.file_uploader("Upload MP3", type=["mp3"])

if uploaded_file:
    st.audio(uploaded_file)

    with st.spinner("Analyzing audio..."):
        y, sr = load_mp3(uploaded_file)
        features = extract_features(y, sr).reshape(1, -1)
        model = load_model()
        prediction = model.predict(features)[0]

    st.success(f"🎼 Predicted Genre: **{prediction}**")
