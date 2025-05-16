import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

# Use same feature extractor from above
from your_feature_module import extract_features  # replace with actual file or copy the function here

# Dummy data generator (replace with your real dataset)
def generate_dummy_data(num_samples=100):
    X = []
    y = []
    genres = ['rock', 'pop', 'jazz', 'hiphop']
    for _ in range(num_samples):
        # simulate random audio feature vectors with fixed size 29
        feat = np.random.rand(29)
        X.append(feat)
        y.append(np.random.choice(genres))
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = generate_dummy_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs("model", exist_ok=True)
    with open("model/genre_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved as model/genre_classifier.pkl")
