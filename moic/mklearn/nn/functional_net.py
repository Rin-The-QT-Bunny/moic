import torch 
import torch.nn as nn
import torch.nn.functional as F

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features,outer_most = None):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            #nn.LayerNorm([out_features]),
            nn.Tanh()
        )
        self.outer_most = outer_most

    def forward(self, input):
        if self.outer_most != None:
            return self.outer_most(self.net(input))
        return self.net(input)

class SPLayer(nn.Module):
    def __init__(self, in_features, out_features):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            #nn.LayerNorm([out_features]),
            nn.Softplus(),
          
        )

    def forward(self, input):
        return self.net(input)
class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=True):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))
            self.net.append(nn.Tanh())

        if outermost_linear:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self,item):
        return self.net[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, input):
        return self.net(input)


class ControlledConvAttention(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,32,5,1,2),
            nn.ReLU(),nn.MaxPool2d(2)
        )
    def forward(self,x,c):
        x = x.to(torch.float32)
        x = self.conv1(x)
        
        c = c.to(torch.float32)
        inp = torch.cat([x,c],-1)
        return x
    

class ConvLayer(nn.Module):
    def __init__(self,width,height,latent):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4,10,5,1,2),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(width/2 * height/2 * 10,latent)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class SIREN(nn.Module):
    def __init__(self,input_dim):
        self.shift = FCBlock(128,3,input_dim,1)
        self.frequency = FCBlock(128,3,input_dim,1)
    def forward(self,x):
        beta = torch.relu(self.shift(x) * 10)
        gammar = torch.relu(self.frequency(x) * 10)
        x = torch.sin(gammar * x + beta)
        return x