import time
import numpy as np
import cv2
from os.path import isfile
import os
from pathlib import Path

DATA = Path("data")

def elapsed_time(start_time):
    return time.time() - start_time

def expand_path_img(path):
    if isfile(os.path.join(DATA/"train_images_circular_1024->256",path)):
        return os.path.join(DATA/"train_images_circular_1024->256",path)
    elif isfile(os.path.join(DATA/"valid_images_circular_1024->256",path)):
        return os.path.join(DATA/"valid_images_circular_1024->256",path)
    elif isfile(os.path.join(DATA/"pseudo_images_circular_1024->256",path)):
        return os.path.join(DATA/"pseudo_images_circular_1024->256",path)
    elif isfile(os.path.join(DATA/"train_images",path)):
        return os.path.join(DATA/"train_images",path)
    elif isfile(os.path.join(DATA/"test_images",path)):
        return os.path.join(DATA/"test_images",path)
#     elif isfile(os.path.join(DATA/"test_images_circular_1024->256",path+'jpg')):
#         return os.path.join(DATA/"test_images_circular_1024->256",path+'jpg')
    return path

def expand_path_label(path, heatmap=False, height=False, width=False):
    
    if heatmap:
        if isfile(os.path.join(DATA/"train_images_heatmaps_circular_1024->256",path)):
            return os.path.join(DATA/"train_images_heatmaps_circular_1024->256",path)
        elif isfile(os.path.join(DATA/"valid_images_heatmaps_circular_1024->256",path)):
            return os.path.join(DATA/"valid_images_heatmaps_circular_1024->256",path)
        elif isfile(os.path.join(DATA/"pseudo_images_heatmaps_circular_1024->256",path)):
            return os.path.join(DATA/"pseudo_images_heatmaps_circular_1024->256",path)
    elif height:
        if isfile(os.path.join(DATA/"train_images_heightmaps_circular_1024->256",path)):
            return os.path.join(DATA/"train_images_heightmaps_circular_1024->256",path)
        elif isfile(os.path.join(DATA/"valid_images_heightmaps_circular_1024->256",path)):
            return os.path.join(DATA/"valid_images_heightmaps_circular_1024->256",path)
        elif isfile(os.path.join(DATA/"pseudo_images_heightmaps_circular_1024->256",path)):
            return os.path.join(DATA/"pseudo_images_heightmaps_circular_1024->256",path)
    
    elif width:
        if isfile(os.path.join(DATA/"train_images_widthmaps_circular_1024->256",path)):
            return os.path.join(DATA/"train_images_widthmaps_circular_1024->256",path)
        elif isfile(os.path.join(DATA/"valid_images_widthmaps_circular_1024->256",path)):
            return os.path.join(DATA/"valid_images_widthmaps_circular_1024->256",path)
        elif isfile(os.path.join(DATA/"pseudo_images_widthmaps_circular_1024->256",path)):
            return os.path.join(DATA/"pseudo_images_widthmaps_circular_1024->256",path)
    
    return path

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)