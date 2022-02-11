import numpy as np
import random
from core.getdata import *
from core.utils import *
from sklearn.preprocessing import StandardScaler

class DatasetNaive(torch.utils.data.Dataset):
    def __init__(self, ligands, replicas, size): 
        self.data = np.empty([0, 98])
        self.labels = np.empty([0])
        
        for i in range(len(ligands)):
            cd = np.empty([0, 98])
            ts = GetData(ligands[i], "", replicas, size)
            cd = np.row_stack((cd, ts))
            self.labels = np.append(self.labels, np.array([i] * len(cd))) 
            self.data = np.row_stack((self.data, cd))
    
        self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], 1)
        self.data = np.float32(self.data) 
        self.labels = np.int64(self.labels) 
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, row):
        return self.data[row], self.labels[row]#, self.polarizations[row]
        #return self.dataset[row]