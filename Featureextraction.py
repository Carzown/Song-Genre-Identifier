import numpy as np
import librosa
import scipy.signal as signal

def low_pass_filter(y, sr, cutoff=4000):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(5, norm_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, y)

def extract_features(y, sr, n_mfcc=13):
    y = low_pass_filter(y, sr)
    
    # Extract features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Aggregate stats: mean for each feature dimension
    features = np.hstack([
        np.mean(chroma, axis=1),      # 12 dims
        np.mean(spec_cent),           # 1 dim
        np.mean(spec_bw),             # 1 dim
        np.mean(rolloff),             # 1 dim
        np.mean(zcr),                 # 1 dim
        np.mean(mfcc, axis=1)         # n_mfcc dims (13 default)
    ])

    # Total features = 12 + 1 + 1 + 1 + 1 + 13 = 29
    return features
