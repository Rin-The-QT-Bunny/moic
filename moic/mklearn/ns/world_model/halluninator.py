import torch
import torch.nn as nn
from aluneth.rinlearn.nn.visual_net import *

class GridWorldHallucinator(nn.Module):
    def __init__(self,resolution,base_dim = 64):
        super().__init__()
        height, width = resolution
        base_dim = 64
        self.Encoder = EncoderNet(height,width,base_dim)
        self.Decoder = DecoderNet(height,width,base_dim)
    
    def forward(self,x):
        code = self.Encoder(x)
        output = self.Decoder(code)
        return output

class GraphWorldHallucinator(nn.Module):
    def __init__(self,semantics_dim = 128):
        super().__init__()
    
    def forward(self,x):
        """
        Input a observed world graph structure and use the graph completion algorithm to complete the world
        complete_N means repeat the single completion for N times.
        """
        return x