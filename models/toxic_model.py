import torch.nn as nn

class ToxicCommentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ToxicCommentModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input.dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def foward(self, x):
        return self.model(x)