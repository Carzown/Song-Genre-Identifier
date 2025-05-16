# train_model.py
import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from dsp_utils import extract_features, low_pass_filter

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
data = []
labels = []

for genre in genres:
    folder = f"genres/{genre}"  # path to dataset
    for filename in os.listdir(folder)[:20]:  # limit for speed
        file_path = os.path.join(folder, filename)
        y, sr = librosa.load(file_path, sr=22050)
        y = low_pass_filter(y, sr)
        features = extract_features(y, sr)
        data.append(features)
        labels.append(genre)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(data, labels)

with open("model/genre_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
