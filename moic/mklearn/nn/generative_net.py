import random
import numpy as np
import torch
import torch.nn as nn
from aluneth.rinlearn.nn.functional_net import *

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        feature_dim = 32
        latent_dim = 6
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim,128),
            nn.ReLU(),
            nn.Linear(128,latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,128),
            nn.ReLU(),
            nn.Linear(128,feature_dim)
        )
    def reparameteriztation(self,mu,logvar):
        eps = random.random()
        z = mu + torch.exp(logvar) * eps
        return z
    
    def forward(self,x):
        encode_params = self.encoder(x)
        #print("encoder params: ",encode_params.shape)
        mu = encode_params[:,:self.latent_dim]
        logvar = encode_params[:,self.latent_dim:]
        #print(mu.shape,logvar.shape)
        latent_z = self.reparameteriztation(mu,logvar)
        #print(latent_z.shape)
        output = self.decoder(latent_z)
        
        kl_loss =  -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return output,kl_loss
    
class GenerativeAdversial(nn.Module):
    def __init__(self,feature_dim,noise_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.Generator = FCBlock(64,1,noise_dim,feature_dim)
        self.Discriminator = FCBlock(64,1,feature_dim,1)

    def observed_data_loss(self,data):
        noise = torch.randn([100,self.noise_dim])
        observed_data = data
        fake_data = self.Generator(noise)
        real_log_prob = torch.log(self.Discriminator(observed_data))
        fake_log_prob = torch.log(1.0 - self.Discriminator(fake_data))        
        
        generator_loss = torch.mean(fake_log_prob)
        discriminate_loss = -torch.mean(fake_log_prob) - torch.mean(real_log_prob)
        return generator_loss, discriminate_loss
    
    def generate(self,num_noise):
        noise = torch.randn([num_noise,self.noise_dim])
        return self.Generator(noise)
