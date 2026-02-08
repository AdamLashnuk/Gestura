import numpy as np
import joblib

class SignClassifier:
    def __init__(self, model_path="model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, landmarks):
        wrist = landmarks[0]
        features = []

        for lm in landmarks:
            features.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

        features = np.array(features).reshape(1, -1)
        return self.model.predict(features)[0]
