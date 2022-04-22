import numpy as np
import pandas as pd
import glob
import torch
import torchvision
import matplotlib.pyplot as plt
import random

class GetData(torch.utils.data.Dataset):
    def __init__(self, ligand, polarization, replicas, size):
        super().__init__()
        
        empty = np.empty([0, 98])
        for i in range(1, replicas + 1):
            loc = 'D:/Data/hoffmanlab/lstmtimeseries/data/' + ligand + polarization + str(i) + '.csv'
            data = pd.read_csv(loc)
            data = data.to_numpy()
            nan_counts = np.sum(np.isnan(data), axis=1)
            nan_rows = np.nonzero(nan_counts)
            data = np.delete(data, nan_rows, axis = 0)
            self.X = np.row_stack((empty, data))
        
        if (len(self.X) > size):
            row_indices = random.choices(range(0, len(self.X)), k = size)
            self.X = self.X[row_indices, :]
        else:
            row_indices = random.choices(range(0, len(self.X)), k = size - len(self.X))
            self.X = np.row_stack((self.X, self.X[row_indices, :]))  
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__ (self, row):
        return self.X[row]
    
class GetReplica(torch.utils.data.Dataset):
    def __init__(self, ligand, polarization, replica, size):
        super().__init__()
        
        empty = np.empty([0, 98])
        
        loc = 'D:/Data/hoffmanlab/lstmtimeseries/data/' + ligand + polarization + str(replica) + '.csv'
        data = pd.read_csv(loc)
        data = data.to_numpy()
        nan_counts = np.sum(np.isnan(data), axis=1)
        nan_rows = np.nonzero(nan_counts)
        data = np.delete(data, nan_rows, axis = 0)
        self.X = np.row_stack((empty, data))
        
        if (len(self.X) > size):
            row_indices = random.choices(range(0, len(self.X)), k = size)
            self.X = self.X[row_indices, :]
        else:
            row_indices = random.choices(range(0, len(self.X)), k = size - len(self.X))
            self.X = np.row_stack((self.X, self.X[row_indices, :]))  
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__ (self, row):
        return self.X[row]
    

        