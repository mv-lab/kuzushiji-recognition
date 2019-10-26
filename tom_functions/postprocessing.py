import numpy as np
import PIL
import skimage.feature
import scipy
import torch
import matplotlib.pyplot as plt
from albumentations import (Compose, Resize, Normalize)
from albumentations.pytorch import ToTensor

import stable.utils
import stable.predict

from tom_functions.preprocessing import *


MEAN1,MEAN2,MEAN3 = 0.485, 0.456, 0.406
STD1,STD2,STD3    = 0.229, 0.224, 0.225


# TODO: detect points in images, no need to re-tile
def get_centers(img, model, device, hm_threshold=20, hm_min_distance=10):
    im = np.asarray(img)
    block_size = 1024
    model_input_size = 256
    channels = 3
    im_height, im_width = im.shape[:2]
    hm_height, hm_width = im_height//2, im_width//2
    
    blocks = stable.utils.get_image_blocks(im, block_size)
    heatmap   = np.zeros([hm_height, hm_width], dtype=np.uint8) #0-255
    heightmap = np.zeros([hm_height, hm_width], dtype=np.uint8) #0-255
    widthmap  = np.zeros([hm_height, hm_width], dtype=np.uint8) #0-255
    
    transform = Compose([
        Resize(height=model_input_size, width=model_input_size,
               interpolation=1, p=1),
        Normalize(mean=(MEAN1, MEAN2, MEAN3), 
                  std=(STD1, STD2, STD3)),
        ToTensor(),
    ])

    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            inputs = transform(image=blocks[i,j])['image'].contiguous()
            c,h,w  = inputs.size()
            inputs = inputs.view(1,c,h,w)
            #logits  = model(inputs.to(device, torch.float32, non_blocking=True)).cpu().detach()
            logits1,logits = model(inputs.to(device, torch.float32, non_blocking=True))
            logits1,logits = logits1.cpu().detach(), logits.cpu().detach()
            hm_tile = torch.sigmoid(logits[:,0,:,:]).squeeze().numpy()
            height_tile = logits[:,1,:,:].squeeze().numpy()
            width_tile  = logits[:,2,:,:].squeeze().numpy()
            #denormalize
            hm_tile = (255*hm_tile).astype(np.uint8)
            height_tile = (255*height_tile).clip(0,255).astype(np.uint8)
            width_tile  = (255*width_tile).clip(0.255).astype(np.uint8)

            if (i+1) * 256 > hm_height:
                hm_tile = hm_tile[:hm_height%256,:]
                height_tile = height_tile[:hm_height%256,:]
                width_tile  = width_tile[:hm_height%256,:]
            if (j+1) * 256 > hm_width:
                hm_tile = hm_tile[:,:hm_width%256]
                height_tile = height_tile[:,:hm_height%256]
                width_tile  = width_tile[:,:hm_height%256]
            
            heatmap[256*i:256*(i+1), 256*j:256*(j+1)] = hm_tile
            heightmap[256*i:256*(i+1), 256*j:256*(j+1)] = height_tile
            widthmap[256*i:256*(i+1), 256*j:256*(j+1)]  = width_tile
            
    # https://stackoverflow.com/questions/51672327/skimage-peak-local-max-finds-multiple-spots-in-close-proximity-due-to-image-impu
    heatmap_gray = np.array(PIL.Image.fromarray(heatmap).convert("L"))
    
    is_peak = skimage.feature.peak_local_max(heatmap_gray, min_distance=hm_min_distance, 
                                             indices=False, threshold_abs=hm_threshold)
    labels = scipy.ndimage.measurements.label(is_peak)[0]
    merged_peaks = scipy.ndimage.measurements.center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
    
    merged_peaks = np.array(merged_peaks)
    outputs = []
    for yc,xc in merged_peaks:
        xc, yc = int(xc), int(yc)
        outputs.append([xc * (block_size / model_input_size), 
                        yc * (block_size / model_input_size), 
                        widthmap[yc,xc],
                        heightmap[yc,xc]])
    return np.array(outputs)


def postprosessing(idx, data_names, model, device):
    data_val = data_names[idx] #[fname+'.jpg' for fname in data_fnames][idx]
    print(data_val)
    img = preprocessing(data_val, test_mode=True) #(h,w,c)

    outputs = get_centers(img, model, device, hm_threshold=20, hm_min_distance=10)
    print('otputs.shape = ', outputs.shape)
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img, cmap="gray")
    ax.plot(outputs[:,0], outputs[:,1], "ro")
    for i in range(len(outputs)):
        ax.add_patch(
            plt.Rectangle((int(outputs[i,0]-outputs[i,2]//2), 
                           int(outputs[i,1]-outputs[i,3]//2)),
            int(outputs[i,2]),
            int(outputs[i,3]), fill=False,
            edgecolor='blue', linewidth=3.5)
        )
    plt.show()
    plt.close()