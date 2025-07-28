import torch
from torch.utils.data import Dataset

# custom pytorch dataset for toxic comment classification task
class ToxicDataset(Dataset):
    def __init__(self, inputs, labels):
        # store into label matrix
        self.inputs = inputs # [num_samples, num_features]
        self.labels = labels # [num_samples, num_labels]

    def __len__(self):
        return len(self.inputs) # return number of samples in dataset

    # get feature and label for sample index
    # convert to pytorch tensors with dtype float32
    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.inputs[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }