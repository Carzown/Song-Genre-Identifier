import numpy as np
import librosa
import scipy.signal as signal
from pydub import AudioSegment
import tempfile

def low_pass_filter(y, sr, cutoff=4000):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(5, norm_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, y)

def extract_features(y, sr):
    y = low_pass_filter(y, sr)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = np.hstack([
        np.mean(chroma, axis=1),
        np.mean(spec_cent),
        np.mean(spec_bw),
        np.mean(rolloff),
        np.mean(zcr),
        np.mean(mfcc, axis=1)
    ])
    return features

def load_mp3(file):
    audio = AudioSegment.from_file(file, format="mp3")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.export(tmp.name, format="wav")
        y, sr = librosa.load(tmp.name, sr=22050)
    return y, sr
