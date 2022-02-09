import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

# dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=True, num_workers=4)
# dic = {"y_pred": np.array([]), "y_true": np.array([])}
# for x_batch, y_batch in dataloader_val:
#     x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
#     # to do : convert y_batch to np array and vstack it y_true
#     y_batch = y_batch.detach().cpu().numpy()
#     dic[y_true].vstack(y_true, y_batch)
#     y_pred = trainer.model(x_batch)
#     y_pred = F.softmax(y_pred, dim=1)
#     # to do : convert to np array and vstack it to y_pred
#     y_pred = y_pred.detach().cpu().numpy()
#     dic[y_pred].vstack(y_pred)
    
# df = pd.DataFrame(dic)