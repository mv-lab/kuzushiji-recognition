from math import ceil
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd


def get_image_blocks(image, block_size):
    height, width, channels = image.shape
    width_blocks = ceil(width / block_size)
    height_blocks = ceil(height / block_size)
    image_blocks = np.empty((height_blocks, width_blocks, block_size, block_size, channels), dtype=np.uint8)
    for i in range(height_blocks):
        for j in range(width_blocks):
            block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            if block.shape != (block_size, block_size, channels):
                # need to pad block
                right_pad = block_size - block.shape[1]
                bottom_pad = block_size - block.shape[0]
                block = np.pad(block, [[0, bottom_pad], [0, right_pad], [0, 0]], "constant", constant_values=0)
            image_blocks[i,j] = block.reshape((1,1,block_size,block_size, channels))
    return image_blocks


def random_valid_split(n_valid=500):
    DATA = Path("data")
    df_train = pd.read_csv(DATA/"train.csv")
    np.random.seed(420)
    train_fnames = list(df_train.image_id)
    valid_fnames = np.random.choice(train_fnames, n_valid, replace=False)
    actual_train_fnames = [f for f in train_fnames if f not in valid_fnames]
    (DATA/"train_fnames.txt").write_text("\n".join(actual_train_fnames))
    (DATA/"valid_fnames.txt").write_text("\n".join(valid_fnames))
    
    
def stratified_valid_split(test_size=0.2):
    from sklearn.model_selection import train_test_split
    DATA = Path("data")
    df_train = pd.read_csv(DATA/"train.csv")
    train_fnames = list(df_train.image_id)
    books = []
    for f in train_fnames:
        books.append(f.split("_")[0].split("-")[0])
    train, valid = train_test_split(train_fnames, test_size=test_size, random_state=420, stratify=books)
    (DATA/"train_fnames.txt").write_text("\n".join(train))
    (DATA/"valid_fnames.txt").write_text("\n".join(valid))
    
    
def group_valid_split(test_size=0.2):
    from sklearn.model_selection import GroupShuffleSplit
    DATA = Path("data")
    df_train = pd.read_csv(DATA/"train.csv")
    train_fnames = np.array(list(df_train.image_id))
    books = []
    for f in train_fnames:
        books.append(f.split("_")[0].split("-")[0])
    group_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=420)
    train_idx, valid_idx = next(group_split.split(train_fnames, groups=books))
    train, valid = train_fnames[train_idx], train_fnames[valid_idx]
    (DATA/"train_fnames.txt").write_text("\n".join(train))
    (DATA/"valid_fnames.txt").write_text("\n".join(valid))