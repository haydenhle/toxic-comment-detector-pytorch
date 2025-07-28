import torch.nn as nn

class ToxicCommentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ToxicCommentModel, self).__init__() # call parent class constructor
        
        # simple feedforward neural network with nn.sequential
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # first layer
            nn.ReLU(), # add non-linearity
            nn.Dropout(0.3), # prevent overfitting (disables 30% of neurons during training randomly)
            nn.Linear(hidden_dim, output_dim), # second layer (6 labels)
            nn.Sigmoid() # squish values from 0 - 1
        )

    # define forward pass
    def forward(self, x):
        return self.model(x) # pass input through network layers above
