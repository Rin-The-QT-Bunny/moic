import torch
import torch.nn as nn

from aluneth.rinlearn.nn.functional_net import *
from aluneth.rinlearn.nn.visual_net import *

class NPModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(sneurelf,x):
        return x