import numpy as np
import random
from core.getdata import *
from core.utils import *
from sklearn.preprocessing import StandardScaler

class Dataset(torch.utils.data.Dataset):
    def __init__(self, ligands, polarizations, replicas, size): #to do: fix dataset classes so it initalizes as float 32
        self.data = np.empty([0, 98])
        self.labels = np.empty([0])
        #self.polarizations = np.empty([0])
        
        for i in range(len(ligands)):
            cd = np.empty([0, 98])
            for j in range(len(polarizations)):
                if (ligands[i] == 'UST'):
                    ts = GetData(ligands[i], "", replicas, size) #1 meaning there are no replicas // could implement **kwargs
                else:
                    ts = GetData(ligands[i], polarizations[j], replicas, size)
                
                #self.polarizations = np.append(self.polarizations, np.array([j] * len(ts))) #len gives number of rows
                cd = np.row_stack((cd, ts))
            self.labels = np.append(self.labels, np.array([i] * len(cd))) 
            self.data = np.row_stack((self.data, cd))
            
        # scaler = StandardScaler() # scaling data in the init
        # self.data = scaler.fit_transform(self.data) #
        self.data = self.data.reshape(self.data.shape[0], self.data.shape[1], 1)
        # self.labels = self.labels.reshape(self.labels.shape[0], 1) 
        self.data = np.float32(self.data) 
        self.labels = np.int64(self.labels) 
            
        #labels = np.reshape(self.labels.T, (self.labels.shape[0], 1))
        #polarizations = np.reshape(self.polarizations.T, (self.polarizations.shape[0], 1))
        #self.dataset = np.concatenate((self.data, labels, polarizations), axis=1)
                                               
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, row):
        return self.data[row], self.labels[row]#, self.polarizations[row]
        #return self.dataset[row]
                                 