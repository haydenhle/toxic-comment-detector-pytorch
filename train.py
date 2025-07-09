import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

import joblib

from models.toxic_model import ToxicCommentModel
from utils.dataset import ToxicDataset

# load data
df = pd.read_csv("data/train.csv")
df = df.fillna("")

# define labels
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# initialize processing
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(df['comment_text']).toarray()
y = df[LABELS].values

# train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

train_ds = ToxicDataset(X_train, y_train)
val_ds = ToxicDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToxicCommentModel(input_dim=10000, hidden_dim=256, output_dim=6).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train loop
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# validate loop
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            preds = (outputs > 0.5).int()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute basic per-label accuracy
    accs = []
    for i in range(OUTPUT_DIM):
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        accs.append(acc)

    print("Validation Accuracy (per label):")
    for label, acc in zip(LABELS, accs):
        print(f"  {label:15}: {acc:.4f}")

# save model and vectorizer
torch.save(model.state_dict(), "model.pth")
joblib.dump(vectorizer, "vectorizer.pkl")
