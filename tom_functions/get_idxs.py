from sklearn.model_selection import KFold
import numpy as np


def get_idxs(data_names):
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    val_idxs_list   = []
    
    for fold, (trn,val) in enumerate(kfold.split(data_names, data_names)):
        val_idxs_list.append(val)
    
    return val_idxs_list