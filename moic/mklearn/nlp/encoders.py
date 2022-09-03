import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self,words_dim,semantics_dim,bilayer = True):
        super().__init__()
        #self.en = nn.TransformerEncoderLayer(word_dim,32)
        self.BiGRU = self.gru = nn.GRU(words_dim,int(semantics_dim/2), batch_first =True, bidirectional=True) 
        self.semantics_dim = semantics_dim
    def forward(self,x):
        len_shape = [1]
        len_shape.extend(x.shape)
        x_t = x.reshape(len_shape)
        y = self.BiGRU(x_t)
        out = torch.sum(y[0][0],0).reshape([1,self.semantics_dim])
        #print(out.shape)
        return out