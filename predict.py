import torch
import torch.nn as nn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from models.toxic_model import ToxicCommentModel

# config
MODEL_PATH = "model.pth"               # path to your saved model
VECTORIZER_PATH = "vectorizer.pkl"     # path to saved TF-IDF vectorizer
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
INPUT_TEXTS = [
    "You are the worst person ever!",
    "I hope you have a great day :)",
    "This is so dumb and offensive"
]

# import TF-IDF vectorizer
import joblib
vectorizer = joblib.load(VECTORIZER_PATH)

# import model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToxicCommentModel(input_dim=10000, hidden_dim=256, output_dim=6).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# make predictions
inputs = torch.tensor(X, dtype=torch.float32).to(device)
with torch.no_grad():
    outputs = model(inputs)

# interpret predictions
predictions = outputs.cpu().numpy()
threshold = 0.5  # Any output > 0.5 is considered a "positive" label

for i, text in enumerate(INPUT_TEXTS):
    print(f"\nComment: {text}")
    for j, label in enumerate(LABELS):
        if predictions[i][j] > threshold:
            print(f"  - {label}: {predictions[i][j]:.2f}")