import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

from core.dataset import *
from core.dataset2 import *
from core.getdata import *
from core.torchdata import *

'''
Returns 60-20-20 split with equal representation across all polarization states
Returns torch.utils.data.Dataset child object

Will optimize this function to take parameters and allow more flexibility in the future if needed
'''
# probably just delete half and just run this function twice...
# pass replica # as parameter
# refactor later if needed
def Preprocess():
    # ligands and polarization 
    ligands = ["TNF", "R84", "PIC", "P3K", "FLA", "CpG", "FSL", "LPS", "UST"]
    polarization = ["", "ib", "ig", "i0", "i3", "i4"]
    size = 1288
    
    # get all data
    m0_1, m0_2 = DatasetSplit(ligands, [""], 1, size), DatasetSplit(ligands, [""], 2, size)
    ib_1, ib_2 = DatasetSplit(ligands, ["ib"], 1, size), DatasetSplit(ligands, ["ib"], 2, size)
    ig_1, ig_2 = DatasetSplit(ligands, ["ig"], 1, size), DatasetSplit(ligands, ["ig"], 2, size)
    i0_1, i0_2 = DatasetSplit(ligands, ["i0"], 1, size), DatasetSplit(ligands, ["i0"], 2, size)
    i3_1, i3_2 = DatasetSplit(ligands, ["i3"], 1, size), DatasetSplit(ligands, ["i3"], 2, size)
    i4_1, i4_2 = DatasetSplit(ligands, ["i4"], 1, size), DatasetSplit(ligands, ["i4"], 2, size)
    
    # 60/20/20 split
    m0_1_train, m0_1_val, m0_1_test = torch.utils.data.random_split(m0_1, [6956, 2318, 2318])
    m0_2_train, m0_2_val, m0_2_test = torch.utils.data.random_split(m0_2, [6956, 2318, 2318])
    ib_1_train, ib_1_val, ib_1_test = torch.utils.data.random_split(ib_1, [6956, 2318, 2318])
    ib_2_train, ib_2_val, ib_2_test = torch.utils.data.random_split(ib_2, [6956, 2318, 2318])
    ig_1_train, ig_1_val, ig_1_test = torch.utils.data.random_split(ig_1, [6956, 2318, 2318])
    ig_2_train, ig_2_val, ig_2_test = torch.utils.data.random_split(ig_2, [6956, 2318, 2318])
    i0_1_train, i0_1_val, i0_1_test = torch.utils.data.random_split(i0_1, [6956, 2318, 2318])
    i0_2_train, i0_2_val, i0_2_test = torch.utils.data.random_split(i0_2, [6956, 2318, 2318])
    i3_1_train, i3_1_val, i3_1_test = torch.utils.data.random_split(i3_1, [6956, 2318, 2318])
    i3_2_train, i3_2_val, i3_2_test = torch.utils.data.random_split(i3_2, [6956, 2318, 2318])
    i4_1_train, i4_1_val, i4_1_test = torch.utils.data.random_split(i4_1, [6956, 2318, 2318])
    i4_2_train, i4_2_val, i4_2_test = torch.utils.data.random_split(i4_2, [6956, 2318, 2318])
    
    # extract data and labels
    m0_1_train_X, m0_1_train_Y, m0_1_val_X, m0_1_val_Y, m0_1_test_X, m0_1_test_Y = extract_xy((m0_1_train, m0_1_val, m0_1_test))
    m0_2_train_X, m0_2_train_Y, m0_2_val_X, m0_2_val_Y, m0_2_test_X, m0_2_test_Y = extract_xy((m0_2_train, m0_2_val, m0_2_test))
    ib_1_train_X, ib_1_train_Y, ib_1_val_X, ib_1_val_Y, ib_1_test_X, ib_1_test_Y = extract_xy((ib_1_train, ib_1_val, ib_1_test))
    ib_2_train_X, ib_2_train_Y, ib_2_val_X, ib_2_val_Y, ib_2_test_X, ib_2_test_Y = extract_xy((ib_2_train, ib_2_val, ib_2_test))
    ig_1_train_X, ig_1_train_Y, ig_1_val_X, ig_1_val_Y, ig_1_test_X, ig_1_test_Y = extract_xy((ig_1_train, ig_1_val, ig_1_test))
    ig_2_train_X, ig_2_train_Y, ig_2_val_X, ig_2_val_Y, ig_2_test_X, ig_2_test_Y = extract_xy((ig_2_train, ig_2_val, ig_2_test))
    i0_1_train_X, i0_1_train_Y, i0_1_val_X, i0_1_val_Y, i0_1_test_X, i0_1_test_Y = extract_xy((i0_1_train, i0_1_val, i0_1_test))
    i0_2_train_X, i0_2_train_Y, i0_2_val_X, i0_2_val_Y, i0_2_test_X, i0_2_test_Y = extract_xy((i0_2_train, i0_2_val, i0_2_test))
    i3_1_train_X, i3_1_train_Y, i3_1_val_X, i3_1_val_Y, i3_1_test_X, i3_1_test_Y = extract_xy((i3_1_train, i3_1_val, i3_1_test))
    i3_2_train_X, i3_2_train_Y, i3_2_val_X, i3_2_val_Y, i3_2_test_X, i3_2_test_Y = extract_xy((i3_2_train, i3_2_val, i3_2_test))
    i4_1_train_X, i4_1_train_Y, i4_1_val_X, i4_1_val_Y, i4_1_test_X, i4_1_test_Y = extract_xy((i4_1_train, i4_1_val, i4_1_test))
    i4_2_train_X, i4_2_train_Y, i4_2_val_X, i4_2_val_Y, i4_2_test_X, i4_2_test_Y = extract_xy((i4_2_train, i4_2_val, i4_2_test))
    
    # replica 1
    train_X_1 = np.vstack([m0_1_train_X, ib_1_train_X, ig_1_train_X, i0_1_train_X, i3_1_train_X, i4_1_train_X])
    train_Y_1 = np.hstack([m0_1_train_Y, ib_1_train_Y, ig_1_train_Y, i0_1_train_Y, i3_1_train_Y, i4_1_train_Y])
    val_X_1 = np.vstack([m0_1_val_X, ib_1_val_X, ig_1_val_X, i0_1_val_X, i3_1_val_X, i4_1_val_X])
    val_Y_1 = np.hstack([m0_1_val_Y, ib_1_val_Y, ig_1_val_Y, i0_1_val_Y, i3_1_val_Y, i4_1_val_Y])
    test_X_1 = np.vstack([m0_1_test_X, ib_1_test_X, ig_1_test_X, i0_1_test_X, i3_1_test_X, i4_1_test_X])
    test_Y_1 = np.hstack([m0_1_test_Y, ib_1_test_Y, ig_1_test_Y, i0_1_test_Y, i3_1_test_Y, i4_1_test_Y])
    
    # replica 2
    train_X_2 = np.vstack([m0_2_train_X, ib_2_train_X, ig_2_train_X, i0_2_train_X, i3_2_train_X, i4_2_train_X])
    train_Y_2 = np.hstack([m0_2_train_Y, ib_2_train_Y, ig_2_train_Y, i0_2_train_Y, i3_2_train_Y, i4_2_train_Y])
    val_X_2 = np.vstack([m0_2_val_X, ib_2_val_X, ig_2_val_X, i0_2_val_X, i3_2_val_X, i4_2_val_X])
    val_Y_2 = np.hstack([m0_2_val_Y, ib_2_val_Y, ig_2_val_Y, i0_2_val_Y, i3_2_val_Y, i4_2_val_Y])
    test_X_2 = np.vstack([m0_2_test_X, ib_2_test_X, ig_2_test_X, i0_2_test_X, i3_2_test_X, i4_2_test_X])
    test_Y_2 = np.hstack([m0_2_test_Y, ib_2_test_Y, ig_2_test_Y, i0_2_test_Y, i3_2_test_Y, i4_2_test_Y])
    
    # concatenate into one dataset
    train1, val1, test1 = TorchData(train_X_1, train_Y_1), TorchData(val_X_1, val_Y_1), TorchData(test_X_1, test_Y_1)
    train2, val2, test2 = TorchData(train_X_2, train_Y_2), TorchData(val_X_2, val_Y_2), TorchData(test_X_2, test_Y_2)
    
    # return
    return train1, val1, test1, train2, val2, test2
   
def extract_xy(subsets):
    train, val, test = subsets # unpack tuple
    train_X = [item[0] for item in train]
    train_Y = [item[1] for item in train]
    val_X = [item[0] for item in val]
    val_Y = [item[1] for item in val]
    test_X = [item[0] for item in test]
    test_Y = [item[1] for item in test]
    
    return train_X, train_Y, val_X, val_Y, test_X, test_Y
    
    