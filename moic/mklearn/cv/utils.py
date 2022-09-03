import torch
from torchvision import transforms

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

def resize(img,resolution): return transforms.Resize(resolution)(img)

def data2images(torch_data): return torch_data.permute([0,2,3,1])

def images2data(images): return torch.tensor(images).permute([0,3,1,2])

def load_image(path,normalize = True):
    """from file path to standard single image data"""
    if normalize: return torch.tensor(np.array(Image.open(path))).float().unsqueeze(0).permute([0,3,1,2])/256
    return torch.tensor(np.array(Image.open(path))).float().unsqueeze(0).permute([0,3,1,2])

def combine_tensors(images):
    output = images2data(images[0:1])
    for i in range(len(images) - 1):
        output = torch.cat([output,images[i+1:i+2]],0)
    return output

def visualize_images(images,cols = 3):
    num_images = images.shape[0]
    rows = num_images / cols
    for i in range(num_images):
        plt.subsplot(rows,cols,i + 1)
        plt.imshow(images[i])
    return 0

from aluneth.utils import *

def render_masks(image_data,name="mask_render",mode=None):
    # (N,4,w,h), (N,1,w,h)
    # (N,w,h,4), (N,w,h,1)
    assert isinstance(mode,tuple),print("Display mode is required")
    N,w,h,c = image_data.shape
    image_data = dnp(image_data)
    plt.figure(name)
    for i in range(N):
        plt.subplot(mode[0],mode[1],i+1);plt.cla()
        plt.imshow(image_data[i])