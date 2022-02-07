import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]