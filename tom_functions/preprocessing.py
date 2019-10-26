import numpy as np
import PIL
from PIL import ImageDraw, ImageFont
from tqdm import tqdm_notebook as tqdm
from fastai.vision import *
from fastai import layers
from math import ceil

import cv2
from tom_functions.utils import *


def preprocessing(path, test_mode=False):
    img = cv2.imread(expand_path_img(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if test_mode:
        return img
    else:
        heatmap   = cv2.imread(expand_path_label(path, heatmap=True))
        heightmap = cv2.imread(expand_path_label(path, height=True))
        widthmap  = cv2.imread(expand_path_label(path, width=True))
        return img, heatmap, heightmap, widthmap
    

def create_sizemap(image_fn, labels, test_mode=False):
    # Convert annotation string to array
    if type(labels) == float:
        labels = np.array([])
    else:
        if not test_mode:
            labels = np.array(labels.split(' ')).reshape(-1, 5)
        elif test_mode:
            labels = np.array(labels.split(' ')).reshape(-1, 3)
    
    img = cv2.imread(expand_path_img(image_fn))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height_map = np.zeros_like(img, dtype=np.uint8)
    width_map  = np.zeros_like(img, dtype=np.uint8)
    
    x_center = None
    y_center = None
    for codepoint, x, y, w, h in labels:
        if not test_mode:
            x, y, w, h = int(x), int(y), int(w), int(h)
        elif test_mode:
            x, y, w, h = int(x), int(y), int(0), int(0)
        x_center = x + w // 2
        y_center = y + h // 2
        
        if (x_center is not None) and (y_center is not None):
            x_r = w//2
            y_r = h//2
            height_map[y_center-y_r:y_center+y_r, x_center-x_r:x_center+x_r] = h
            width_map[y_center-y_r:y_center+y_r, x_center-x_r:x_center+x_r] = w
                
    return height_map, width_map


def generate_heightmap_crops_circular(im_path, hm_crop_save_path, df, overlap=256, crop_size=1024, resize_to=256, radius=25, test_mode=False):
    if not hm_crop_save_path.exists(): hm_crop_save_path.mkdir()

    for row in tqdm(df.values):
        img_id, labels = row
        hm, _ = create_sizemap("{}.jpg".format(img_id), labels, test_mode=test_mode) #heightmap
        im = np.array(PIL.Image.open(im_path/f"{img_id}.jpg"))
        height, width = im.shape[:2]
        
        for x_i in range(ceil((width - crop_size) / overlap)): 
            # ceil because we want to include crops with the edge of the image
            for y_i in range(ceil((height - crop_size) / overlap)):
                # Note that the crops which include the right or bottom edges might be less than crop_size x crop_size
                x_offset = x_i * overlap
                y_offset = y_i * overlap
                
                cropped_im = im[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
                cropped_hm = hm[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
                
#                 #delete edges
#                 mask = np.zeros_like(cropped_hm)
#                 h,w = mask.shape[:2]
#                 mask[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = 1
#                 cropped_hm = cropped_hm * mask
                
                #save
                fname = f"{img_id}___{str(x_i).zfill(3)}_{str(y_i).zfill(3)}.png"
                PIL.Image.fromarray(cropped_hm).resize([resize_to, resize_to]).save(hm_crop_save_path/fname)
                
                
def generate_widthmap_crops_circular(im_path, hm_crop_save_path, df, overlap=256, crop_size=1024, resize_to=256, radius=25, test_mode=False):
    if not hm_crop_save_path.exists(): hm_crop_save_path.mkdir()

    for row in tqdm(df.values):
        img_id, labels = row
        _, hm = create_sizemap("{}.jpg".format(img_id), labels, test_mode=test_mode) #widthmap
        im = np.array(PIL.Image.open(im_path/f"{img_id}.jpg"))
        height, width = im.shape[:2]
        
        for x_i in range(ceil((width - crop_size) / overlap)): 
            # ceil because we want to include crops with the edge of the image
            for y_i in range(ceil((height - crop_size) / overlap)):
                # Note that the crops which include the right or bottom edges might be less than crop_size x crop_size
                x_offset = x_i * overlap
                y_offset = y_i * overlap
                
                cropped_im = im[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
                cropped_hm = hm[y_offset:y_offset+crop_size, x_offset:x_offset+crop_size]
                
#                 #delete edges
#                 mask = np.zeros_like(cropped_hm)
#                 h,w = mask.shape[:2]
#                 mask[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = 1
#                 cropped_hm = cropped_hm * mask
                
                #save
                fname = f"{img_id}___{str(x_i).zfill(3)}_{str(y_i).zfill(3)}.png"
                PIL.Image.fromarray(cropped_hm).resize([resize_to, resize_to]).save(hm_crop_save_path/fname)