# import libraries
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import joblib

# import model and dataset
from models.toxic_model import ToxicCommentModel
from utils.dataset import ToxicDataset

# initialize config/const
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
OUTPUT_DIM = len(LABELS)
INPUT_DIM = 10000
HIDDEN_DIM = 256
MODEL_PATH = "model.pth"
VECTORIZER_PATH = "vectorizer.pkl"

# load and clean dataset 
df = pd.read_csv("data/train.csv") # load
df = df.fillna("") # replace NaNs with empty strings

# define labels
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# vectorize comment text with TF-IDF
vectorizer = TfidfVectorizer(max_features=INPUT_DIM, stop_words='english') # take top 10000 tokens
X = vectorizer.fit_transform(df['comment_text']).toarray() # turn comments to number arrays
y = df[LABELS].values # extract label columns as target matrix

# train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42) # split into train and validation sets

# wrap data with pytorch dataset classes
train_ds = ToxicDataset(X_train, y_train)
val_ds = ToxicDataset(X_val, y_val)

# create pytorch dataloaders for batching
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ToxicCommentModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(device)

# loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train loop
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train() # start training mode
    train_loss = 0 # track loss

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        inputs = batch['input'].to(device) # input vectors
        labels = batch['label'].to(device) # label vectors

        optimizer.zero_grad()
        outputs = model(inputs) # forward pass
        loss = criterion(outputs, labels) # compute loss
        loss.backward() 
        optimizer.step() # update weights

        train_loss += loss.item() # add training loss

    # calc avg loss
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

    # validate loop
    model.eval() # start eval mode
    all_preds = []
    all_labels = []

    # disable gradient tracking for validation
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            preds = (outputs > 0.5).int() # convert probabilities to binary

            all_preds.append(preds.cpu()) # collect predictions
            all_labels.append(labels.cpu()) # collect labels

    # put predictions and labels together from all batches
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # find basic per-label accuracy
    accs = []
    for i in range(OUTPUT_DIM):
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        accs.append(acc)

    print("Validation Accuracy (per label):")
    for label, acc in zip(LABELS, accs):
        print(f"  {label:15}: {acc:.4f}")

# save model and vectorizer
torch.save(model.state_dict(), MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"\nModel saved to {MODEL_PATH}")
print(f"Vectorizer saved to {VECTORIZER_PATH}")
