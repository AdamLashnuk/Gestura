import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = pd.read_csv("data/asl_letters.csv")

X = data.iloc[:, :-1]   # landmark features
y = data.iloc[:, -1]    # labels

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")
