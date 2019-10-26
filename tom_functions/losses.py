import torch
import numpy as np


def dice_loss(logits, target):
    smooth = 1.
    prob  = torch.sigmoid(logits)
    batch = prob.size(0)
    
    prob   = prob.view(batch,4,-1)
    target = target.view(batch,4,-1)
    
    intersection = torch.sum(prob*target, dim=2)
    denominator  = torch.sum(prob, dim=2) + torch.sum(target, dim=2)
    
    dice = (2*intersection + smooth) / (denominator + smooth)
    dice = torch.mean(dice)
    dice_loss = 1. - dice
    return dice_loss


def dice_sum(img, mask, config):
    batch = img.shape[0]
    
    #flatten
    img  = img.reshape(batch,-1)
    mask = mask.reshape(batch,-1)
    
    dice_array = np.zeros(batch)
    for i in range(batch):
        img_i   = (img[i,:]>config['dice_threshold']).astype(np.float32)
        mask_i  = (mask[i,:]>config['dice_threshold']).astype(np.float32)
        dice_array[i] = 2*np.sum(img_i*mask_i) / (np.sum(img_i) + np.sum(mask_i) + 1e-12)
    return dice_array.sum()