# import libraries
import torch
import torch.nn as nn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# import model
from models.toxic_model import ToxicCommentModel

# config
MODEL_PATH = "model.pth" # path to model weights
VECTORIZER_PATH = "vectorizer.pkl" # path to TF-IDF vectorizer
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] # output labels
# sample test phrases
INPUT_TEXTS = [
    "You are the worst person ever!",
    "I hope you have a great day :)",
    "This is so dumb and offensive"
]

# load TF-IDF vectorizer (limit 10000)
vectorizer = joblib.load(VECTORIZER_PATH)

# convert input texts to vectors
X = vectorizer.transform(INPUT_TEXTS).toarray()

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToxicCommentModel(input_dim=10000, hidden_dim=256, output_dim=6).to(device) # recreate model struct
model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # load trained weights
model.eval() # set mode to eval

# make predictions
inputs = torch.tensor(X, dtype=torch.float32).to(device) # convert input to tensor
with torch.no_grad():
    outputs = model(inputs) # run forward pass

# interpret predictions
predictions = outputs.cpu().numpy() # move predictions to CPU and convert to NumPy
threshold = 0.5

# print predictions
for i, text in enumerate(INPUT_TEXTS):
    print(f"\nComment: {text}")
    for j, label in enumerate(LABELS):
        if predictions[i][j] > threshold:
            print(f"  - {label}: {predictions[i][j]:.2f}")
