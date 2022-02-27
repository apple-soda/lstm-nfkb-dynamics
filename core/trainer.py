import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import tqdm 
import pickle
import numpy as np
import sklearn.metrics

class LSTMTrainer:
    def __init__(self, model, lr=1e-3, device="cpu"):
        self.optim = optim.Adam(model.parameters(), lr=lr)
        self.device = torch.device(device)
        self.network = model
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.loss_history = [[],[]]
        self.kfcv_histories = []
    
    def train_step(self, x, y):
        self.optim.zero_grad()
        y_pred = self.network(x)
        loss = self.loss_fn(y_pred, y) 
        loss.backward()
        self.optim.step()
        return loss
    
    def validate_step(self, x, y):
        y_pred = self.network(x)
        loss = self.loss_fn(y_pred, y) 
        return loss
    
    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.network.load_state_dict(checkpoint['network'])
        self.optim.load_state_dict(checkpoint['optim'])
    
    def save(self, path): #fix all this code later lol
        torch.save({
            'network': self.network.state_dict(),
            'optim': self.optim.state_dict(),
            'loss_history': self.loss_history,
        }, path)
    
    def train(self, dataloader_train, dataloader_val, batch_size=64, n_epochs=50):
        """
        Will be implementing list-dataloaders soon
        """
        torchtype = isinstance(dataloader_train, list)
        if torchtype is False:
            
            for epoch in tqdm.tqdm(range(1, n_epochs + 1)):
                batch_losses = []
                for x_batch, y_batch in dataloader_train: 
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    loss = self.train_step(x_batch, y_batch)
                    loss = float(loss.cpu().detach())
                    batch_losses.append(loss)
                epoch_loss = np.mean(batch_losses)
                self.loss_history[0].append(epoch_loss)  

                val_losses = []
                for x_batch, y_batch in dataloader_val:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    loss = self.validate_step(x_batch, y_batch)
                    loss = float(loss.cpu().detach())
                    val_losses.append(loss)
                vpoch_loss = np.mean(val_losses)
                self.loss_history[1].append(vpoch_loss)

                print(f'Epoch {epoch+0:03}: | Training Loss: {epoch_loss} | Validation Loss: {vpoch_loss}')
    
    def kfcv(self, dataset, k, save_dir, batch_size=65, n_epochs=50):
        """
        Function for k-fold cross validation.
        Length of k_loss must equal k, returns a list of the loss history instead of storing it in self.
        Only optimized for 5-fold cross validation as of right now since I need to implement some algorithm to set the batch sizes and split the dataset accordingly.
        """
        
        d1, d2, d3, d4, d5, remainder = torch.utils.data.random_split(dataset, [13910, 13910, 13910, 13910, 13910, 2]) #remainder can be discarded
        for i in range(k):
            self.kfcv_histories.append([])
        
        """
        CV 1
        """
        self.load('../models/empty.pth')
        dataset_train = torch.utils.data.ConcatDataset([d2, d3, d4, d5])
        dataset_val = d1
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=65, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=65, shuffle=True)
        self.train(dataloader_train, dataloader_val, batch_size=batch_size, n_epochs=n_epochs)
        self.kfcv_histories[0] = self.loss_history
        self.loss_history = [[], []] 
        self.save('../models/cfv/' + save_dir + '1' + '.pth')
        """
        CV 2
        """
        self.load('../models/empty.pth')
        dataset_train = torch.utils.data.ConcatDataset([d1, d3, d4, d5])
        dataset_val = d2
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=65, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=65, shuffle=True)
        self.train(dataloader_train, dataloader_val, batch_size=batch_size, n_epochs=n_epochs)
        self.kfcv_histories[1] = self.loss_history
        self.loss_history = [[], []] 
        self.save('../models/cfv/' + save_dir + '2' + '.pth')
        """
        CV 3
        """
        self.load('../models/empty.pth')
        dataset_train = torch.utils.data.ConcatDataset([d1, d2, d4, d5])
        dataset_val = d3
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=65, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=65, shuffle=True)
        self.train(dataloader_train, dataloader_val, batch_size=batch_size, n_epochs=n_epochs)
        self.kfcv_histories[2] = self.loss_history
        self.loss_history = [[], []] 
        self.save('../models/cfv/' + save_dir + '3' + '.pth')
        """
        CV 4
        """
        self.load('../models/empty.pth')
        dataset_train = torch.utils.data.ConcatDataset([d1, d2, d3, d5])
        dataset_val = d4
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=65, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=65, shuffle=True)
        self.train(dataloader_train, dataloader_val, batch_size=batch_size, n_epochs=n_epochs)
        self.kfcv_histories[3] = self.loss_history
        self.loss_history = [[], []] 
        self.save('../models/cfv/' + save_dir + '4' + '.pth')
        """
        CV 5
        """
        self.load('../models/empty.pth')
        dataset_train = torch.utils.data.ConcatDataset([d1, d2, d3, d4])
        dataset_val = d5
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=65, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=65, shuffle=True)
        self.train(dataloader_train, dataloader_val, batch_size=batch_size, n_epochs=n_epochs)
        self.kfcv_histories[4] = self.loss_history
        self.loss_history = [[], []] 
        self.save('../models/cfv/' + save_dir + '5' + '.pth')
        
        
    
        