import torch
import torch.nn as nn
import torch.nn.functional as F

class CelebAVAE(nn.Module):
    def __init__(self, latent_space=256, input_size=(1, 64, 64)):
        super(CelebAVAE, self).__init__() # call super constructor
        
        self.latent_space = latent_space
        d0, d1, d2 = input_size
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2

        encoder_layers = []
        decoder_layers = []

        # (3,d,d)
        encoder_layers.append(nn.Conv2d(d0, 64, kernel_size=3, stride=1, padding=1))
        encoder_layers.append(nn.LeakyReLU())
        encoder_layers.append(nn.MaxPool2d(2))
        encoder_layers.append(nn.BatchNorm2d(64))
        # (32,d//2,d//2)
        encoder_layers.append(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        encoder_layers.append(nn.LeakyReLU())
        encoder_layers.append(nn.MaxPool2d(2))
        encoder_layers.append(nn.BatchNorm2d(128))
        # (64,d//4,d//4)
        encoder_layers.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        encoder_layers.append(nn.LeakyReLU())
        # (128,d//4,d//4)
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc0 = nn.Linear(128*(d1//4)*(d2//4), 2048)
        # (2048)
        self.mu = nn.Linear(2048, latent_space)
        self.logvar = nn.Linear(2048, latent_space)
        self.dec = nn.Linear(latent_space, 2048)
        
        # (2048)
        self.fc1 = nn.Linear(2048, 128*(d1//4)*(d2//4))
        # (128,d//4,d//4)
        decoder_layers.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.UpsamplingBilinear2d(size=(d1//2,d2//2)))
        encoder_layers.append(nn.BatchNorm2d(128))
        # (64,d//2,d//2)
        decoder_layers.append(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.UpsamplingBilinear2d(size=(d1,d2)))
        encoder_layers.append(nn.BatchNorm2d(64))
        # (32,d,d)
        decoder_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.Conv2d(64, d0, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.Sigmoid())
        # (3,d,d)
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, 128*(self.d1//4)*(self.d2//4))
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
        h = h.view(-1,128, self.d1//4, self.d2//4)
        return self.decoder(h)
    
    def forward(self, x):  # implements the entire feed-forward network.
        mu, logvar = self.encode(x)  # encode
        z = self.reparameterize(mu, logvar)        # sample latent variable
        return self.decode(z), mu, logvar, z          # decode, return
