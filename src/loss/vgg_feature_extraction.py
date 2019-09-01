import torch
import torchvision
import torch.nn as nn

class VggFeatureExtractor(nn.Module):
    def __init__(self):
        super(VggFeatureExtractor, self).__init__()
        
        # download vgg
        vgg16 = torchvision.models.vgg16(pretrained=True).features
        
        # set non trainable
        for param in vgg16.parameters():
            param.requires_grad = False
        
        # slice model
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(4): # conv relu conv relu
            self.slice1.add_module(str(x), vgg16[x])
        for x in range(4, 9): # max conv relu conv relu 
            self.slice2.add_module(str(x), vgg16[x])
        for x in range(9, 16): # max cov relu conv relu conv relu
            self.slice3.add_module(str(x), vgg16[x])
        for x in range(16, 23): # conv relu max conv relu conv relu
            self.slice4.add_module(str(x), vgg16[x])
        for x in range(23, 30): # conv relu conv relu max conv relu
            self.slice5.add_module(str(x), vgg16[x])

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]