import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, device="cpu"): # can only use dropout with input_size > 1
        super(LSTM, self).__init__()
        
        self.input_size = input_size # H0
        self.hidden_size = hidden_size # H1
        self.num_layers = num_layers
        self.device = torch.device(device)

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
        self.to(self.device)
        
    def forward(self, x):
        # import pdb; pdb.set_trace()
        # input must be (N,L,H) because batch first
        # Initializing hidden state and cell state for first input with zeros 
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, dtype=torch.float32).requires_grad_() # num layers, B (batch size), H1 (hidden size)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, dtype=torch.float32).requires_grad_()
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        # import pdb; pdb.set_trace()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
        
