Toxic Comment Detector using PyTorch
- A multi-label text classification project that detects toxic comments using a neural network trained on the Jigsaw Toxic Comment Classification dataset.

Features:
- Multi-label classification for 6 types of toxicity
  - toxic
  - severe_toxic
  - obscene
  - threat
  - insult
  - identity_hate
- TF-IDF vectorization of comment text
- Two-layer neural network using PyTorch
- Training and validation
- Model and vectorizer saving/loading
- Inference with 'predict.py'

toxic-comment-detector-pytorch/
├── data/
│ └── train.csv # Dataset (Jigsaw)
├── models/
│ └── toxic_model.py # PyTorch model definition
├── utils/
│ └── dataset.py # Custom PyTorch Dataset class
├── model.pth # Saved model weights
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── train.py # Training script
├── predict.py # Prediction script
├── requirements.txt # Dependencies
└── README.md # Project documentation

Getting Started
1. Clone repository
   git clone https://github.com/haydenhle/toxic-comment-detector-pytorch.git
2. Install Dependencies
   pip install -r requirements.txt
3. Prepare Dataset https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
   place train.csv from Jigsaw dataset on Kaggle into data folder
4. Training
   python train.py
5. Prediction
   python predict.py

Example Output:
Comment: You are the worst person ever!
  - toxic: 0.92
  - insult: 0.85

Comment: I hope you have a great day :)
  (No toxic labels detected)