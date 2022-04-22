import torch
import torch.nn as nn
from torch.data.utils import Dataset

class TorchData(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx] 