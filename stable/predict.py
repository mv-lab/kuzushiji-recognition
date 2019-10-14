import skimage.feature
import scipy.ndimage.measurements
from fastai.vision import *
import numpy as np


def preprocess_input(arr):
    return Image(torch.Tensor(np.transpose(arr, [2, 0, 1])/255))


def process_output(output):
    return np.array(np.transpose(output[0].data, [1, 2, 0]) * 255, dtype=np.uint8)