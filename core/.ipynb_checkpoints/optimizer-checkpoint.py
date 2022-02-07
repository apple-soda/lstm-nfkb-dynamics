import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, X, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        y_hat = self.model(X)

        # Computes loss
        loss = self.loss_fn(y, y_hat)
 
        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss and accuracy
        return loss.item()
    
    def train(self, device, dataloader_train, batch_size=64, n_epochs=50, n_features=1):
        #model_path = f'models/{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for X_batch, y_batch in dataloader_train:
                
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                loss = self.train_step(X_batch, y_batch)
                batch_losses.append(loss)
                
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            
            #with torch.no_grad():
                #batch_val_losses = []
                #for x_val, y_val in dataloader_val:
                    #x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    #y_val = y_val.to(device)
                    #self.model.eval()
                    #yhat = self.model(x_val)
                    #val_loss = self.loss_fn(y_val, yhat).item()
                    #batch_val_losses.append(val_loss)
                #validation_loss = np.mean(batch_val_losses)
                #self.val_losses.append(validation_loss)

        print (f'Epoch {e+0:03}: | Loss: {training_loss/len(dataloader_train):.5f}')

        #torch.save(self.model.state_dict(), model_path)