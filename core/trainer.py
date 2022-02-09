import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm 
import numpy as np
import sklearn.metrics

class LSTMTrainer:
    def __init__(self, model, lr=1e-3, device="cpu"):
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.device = torch.device(device)
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y) 
        loss.backward()
        self.optimizer.step()
        return loss
    
    def validate_step(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y) 
        return loss
    
    def train(self, dataloader_train, dataloader_val, batch_size=64, n_epochs=50):
        for epoch in tqdm.tqdm(range(1, n_epochs + 1)):
            batch_losses = []
            for x_batch, y_batch in dataloader_train: 
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                loss = self.train_step(x_batch, y_batch)
                loss = float(loss.cpu().detach())
                batch_losses.append(loss)
            epoch_loss = np.mean(batch_losses)
            self.train_losses.append(epoch_loss)  
        
            val_losses = []
            for x_batch, y_batch in dataloader_val:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                loss = self.validate_step(x_batch, y_batch)
                loss = float(loss.cpu().detach())
                val_losses.append(loss)
            vpoch_loss = np.mean(val_losses)
            self.val_losses.append(vpoch_loss)
            
            print(f'Epoch {epoch+0:03}: | Training Loss: {epoch_loss} | Validation Loss: {vpoch_loss}')
        