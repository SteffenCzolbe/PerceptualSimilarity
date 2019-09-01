import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistVAE(nn.Module):
    def __init__(self, latent_space=2, input_size=(32, 32)):
        super(MnistVAE, self).__init__() # call super constructor
        
        self.latent_space = latent_space
        d0, d1 = input_size
        self.d0 = d0
        self.d1 = d1

        encoder_layers = []
        decoder_layers = []

        # (1,d,d)
        encoder_layers.append(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1))
        encoder_layers.append(nn.LeakyReLU())
        encoder_layers.append(nn.MaxPool2d(2))
        # (32,d//2,d//2)
        encoder_layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        encoder_layers.append(nn.LeakyReLU())
        # (64,d//2,d//2)
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc0 = nn.Linear(64*(d0//2)*(d1//2), 1024)
        # (1024)
        self.mu = nn.Linear(1024, latent_space)
        self.logvar = nn.Linear(1024, latent_space)
        self.dec = nn.Linear(latent_space, 1024)
        
        # (1024)
        self.fc1 = nn.Linear(1024, 64*(d0//2)*(d1//2))
        # (64,d//2,d//2)
        decoder_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.UpsamplingBilinear2d(size=(d0,d1)))
        # (64,d,d)
        decoder_layers.append(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.Sigmoid())
        # (1,d,d)
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, 64*(self.d0//2)*(self.d1//2))
        h = F.relu(self.fc0(h)) 
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar): # reparameterization trick
        std = torch.exp(0.5*logvar) # calculates std
        eps = torch.randn_like(std) # samples an epsilon
        assert not torch.isnan(std).any(), 'NAN-Values during reparameterization!'
        assert not torch.isinf(std).any(), 'Infinity-Values during reparameterization!'
        return eps.mul(std).add_(mu) # returns sample as if drawn from mu, std

    def decode(self, z):
        h = F.relu(self.dec(z))
        h = F.relu(self.fc1(h))
        h = h.view(-1,64, self.d0//2, self.d1//2)
        return self.decoder(h)
    
    def forward(self, x):  # implements the entire feed-forward network.
        mu, logvar = self.encode(x)  # encode
        z = self.reparameterize(mu, logvar)        # sample latent variable
        return self.decode(z), mu, logvar, z          # decode, return
