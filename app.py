import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# Dummy training data — replace with real feature vectors
X = np.random.rand(100, 32)  # 100 samples, 32 features (like from extract_features)
y = np.random.choice(['rock', 'pop', 'jazz', 'hiphop'], size=100)

model = RandomForestClassifier()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
with open("model/genre_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved to model/genre_classifier.pkl")
