import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        