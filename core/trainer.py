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
        # need to ensure everything is on the same device
        self.device = device
        self.network = model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.loss_history = [[],[]]
        self.optim = optim.Adam(self.network.parameters(), lr=lr)
        
        """
        Probabilities, y_pred, y_true, etc initalized in respective functions
        """
        
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
                
    def evaluate(self, dataloader_val, batch_size=64, probability_size=9):
      
        self.y_prob = np.empty([batch_size, probability_size])
        self.y_pred = np.empty([batch_size, ])
        self.y_true = np.empty([batch_size, ])
        
        for x, y in dataloader_val:
            x, y = x.to(self.device), y.to(self.device)
            y_true = y.detach().cpu().numpy()
            self.y_true = np.hstack([self.y_true, y_true.T])
            y_pred = self.network(x)
            y_pred = F.softmax(y_pred, dim=1)
            y_pred = y_pred.detach().cpu().numpy()
            self.y_prob = np.vstack([self.y_prob, y_pred])
            y_pred = np.argmax(y_pred, axis=1)
            self.y_pred = np.hstack([self.y_pred, y_pred.T])
    
        self.y_prob = self.y_prob[batch_size:]
        self.y_pred = self.y_pred[batch_size:]
        self.y_true = self.y_true[batch_size:]
        
        return self.y_prob, self.y_pred, self.y_true
        
    def kfcv(self, dataset, k, path, save_dir, batch_size=65, n_epochs=50):
        """
        Function for k-fold cross validation.
        Length of k_loss must equal k, returns a list of the loss history instead of storing it in self.
        Only optimized for 5-fold cross validation as of right now since I need to implement some algorithm to set the batch sizes and split the dataset accordingly.
        """
        self.kfcv_histories = []
        self.kfcv_prob = []
        self.kfcv_pred = []
        self.kfcv_true = []
        d1, d2, d3, d4, d5, remainder = torch.utils.data.random_split(dataset, [13910, 13910, 13910, 13910, 13910, 2]) #remainder can be discarded
        
        for i in range(k):
            self.kfcv_histories.append([])
            self.kfcv_prob.append([])
            self.kfcv_pred.append([])
            self.kfcv_true.append([])
    
        """
        CV 1
        """
        self.load(path)
        dataset_train = torch.utils.data.ConcatDataset([d2, d3, d4, d5])
        dataset_val = d1
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=65, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=65, shuffle=True)
        self.train(dataloader_train, dataloader_val, batch_size=batch_size, n_epochs=n_epochs)
        self.kfcv_histories[0] = self.loss_history
        self.kfcv_prob[0], self.kfcv_pred[0], self.kfcv_true[0] = self.evaluate(dataloader_val, batch_size=65, probability_size=9)
        self.loss_history = [[], []] 
        self.save('../models/cfv/' + save_dir + '1' + '.pth')
        """
        CV 2
        """
        self.load(path)
        dataset_train = torch.utils.data.ConcatDataset([d1, d3, d4, d5])
        dataset_val = d2
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=65, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=65, shuffle=True)
        self.train(dataloader_train, dataloader_val, batch_size=batch_size, n_epochs=n_epochs)
        self.kfcv_histories[1] = self.loss_history
        self.kfcv_prob[1], self.kfcv_pred[1], self.kfcv_true[1] = self.evaluate(dataloader_val, batch_size=65, probability_size=9)
        self.loss_history = [[], []] 
        self.save('../models/cfv/' + save_dir + '2' + '.pth')
        """
        CV 3
        """
        self.load(path)
        dataset_train = torch.utils.data.ConcatDataset([d1, d2, d4, d5])
        dataset_val = d3
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=65, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=65, shuffle=True)
        self.train(dataloader_train, dataloader_val, batch_size=batch_size, n_epochs=n_epochs)
        self.kfcv_histories[2] = self.loss_history
        self.kfcv_prob[2], self.kfcv_pred[2], self.kfcv_true[2] = self.evaluate(dataloader_val, batch_size=65, probability_size=9)
        self.loss_history = [[], []] 
        self.save('../models/cfv/' + save_dir + '3' + '.pth')
        """
        CV 4
        """
        self.load(path)
        dataset_train = torch.utils.data.ConcatDataset([d1, d2, d3, d5])
        dataset_val = d4
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=65, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=65, shuffle=True)
        self.train(dataloader_train, dataloader_val, batch_size=batch_size, n_epochs=n_epochs)
        self.kfcv_histories[3] = self.loss_history
        self.kfcv_prob[3], self.kfcv_pred[3], self.kfcv_true[3] = self.evaluate(dataloader_val, batch_size=65, probability_size=9)
        self.loss_history = [[], []] 
        self.save('../models/cfv/' + save_dir + '4' + '.pth')
        """
        CV 5
        """
        self.load(path)
        dataset_train = torch.utils.data.ConcatDataset([d1, d2, d3, d4])
        dataset_val = d5
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=65, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=65, shuffle=True)
        self.train(dataloader_train, dataloader_val, batch_size=batch_size, n_epochs=n_epochs)
        self.kfcv_histories[4] = self.loss_history
        self.kfcv_prob[4], self.kfcv_pred[4], self.kfcv_true[4] = self.evaluate(dataloader_val, batch_size=65, probability_size=9)
        self.loss_history = [[], []] 
        self.save('../models/cfv/' + save_dir + '5' + '.pth')
        
        
    
        