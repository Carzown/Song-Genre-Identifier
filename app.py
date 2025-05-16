import streamlit as st
import numpy as np
import pickle
import librosa
import scipy.signal as signal
import tempfile

# --- Include extract_features and low_pass_filter here exactly as above ---

def low_pass_filter(y, sr, cutoff=4000):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(5, norm_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, y)

def extract_features(y, sr, n_mfcc=13):
    y = low_pass_filter(y, sr)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    features = np.hstack([
        np.mean(chroma, axis=1),
        np.mean(spec_cent),
        np.mean(spec_bw),
        np.mean(rolloff),
        np.mean(zcr),
        np.mean(mfcc, axis=1)
    ])

    return features

def load_audio(file, duration_sec=20):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    y, sr = librosa.load(tmp_path, sr=22050, duration=duration_sec)
    return y, sr

@st.cache_resource
def load_model():
    with open("model/genre_classifier.pkl", "rb") as f:
        return pickle.load(f)

st.title("🎧 Genre Identifier (First 20 Seconds)")

st.markdown("""
Upload an `.mp3` file. The app analyzes only the first 20 seconds and predicts the music genre.
""")

uploaded_file = st.file_uploader("Upload MP3 File", type=["mp3"])

if uploaded_file:
    st.audio(uploaded_file)

    with st.spinner("Analyzing first 20 seconds..."):
        y, sr = load_audio(uploaded_file, duration_sec=20)
        features = extract_features(y, sr).reshape(1, -1)

        st.write(f"Feature vector shape: {features.shape}")

        model = load_model()
        prediction = model.predict(features)[0]

    st.success(f"🎼 Predicted Genre: **{prediction}**")
