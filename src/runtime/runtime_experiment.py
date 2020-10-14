#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm as tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import pickle
from IPython.core.debugger import set_trace
import sys
import PIL
from PIL import Image
from matplotlib import pyplot as plt


# load loss functions
sys.path.append('../loss')
from loss_provider import LossProvider


# In[2]:


dataset_path = '../datasets/celebA/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
data_dim = (3,64,64)
data_size = np.prod(data_dim)
batch_size = 128


# In[3]:



# key word args for loading data
kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}

# transformers
transformers = transforms.Compose([
   transforms.ToTensor()                                # as tensors
])
transformers_la = transforms.Compose([
   transforms.Grayscale(),
   transforms.ToTensor()                                # as tensors
])

data_set = datasets.ImageFolder(dataset_path, transform=transformers)
data_set_la = datasets.ImageFolder(dataset_path, transform=transformers_la)


# load datasets and make them easily fetchable in DataLoaders
data_loader = torch.utils.data.DataLoader(
   data_set,
   batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
data_loader_la = torch.utils.data.DataLoader(
   data_set_la,
   batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)


# # Data loading

# In[4]:


for data, lable in data_loader:
    x = data
    break
    
for data, lable in data_loader_la:
    x_la = data
    break
    
x = x.to(device)
x_la = x_la.to(device)


# # We reconstruct an inut sample, for 500 iterations

# In[ ]:


class ReconSample(nn.Module):
    def __init__(self, ground_truth, loss_function):
        super().__init__()
        self.loss = loss_function
        self.recon = nn.Parameter(torch.randn(ground_truth.shape))
        self.sigmoid = nn.Sigmoid()
        
    def get_recon(self):
        return self.sigmoid(self.recon)
    
    def forward(self, ground_truth):
        return self.loss(self.get_recon(), ground_truth)

def runtime_test(x, loss_function, epochs=500):
    reconstructor = ReconSample(x, loss_function)
    reconstructor = reconstructor.to(device)
    optimizer = torch.optim.SGD(reconstructor.parameters(), lr=10**-4)
    
    # train
    torch.cuda.reset_max_memory_allocated()
    mem0 =  torch.cuda.max_memory_allocated() 
    reconstructor.loss = reconstructor.loss.to(device)
    t0 = time.time()
    for iter in tqdm(range(epochs), leave=True, position=0):
        optimizer.zero_grad()
        loss = reconstructor.forward(x)
        loss.backward()
        optimizer.step()
    t1 = time.time()
    mem1 = torch.cuda.max_memory_allocated()
        
    return {'runtime':t1 - t0, 'memory':(mem1 - mem0) / (1024**2)}
        


# # Run test for each loss function.

# In[9]:


loss_provider = LossProvider()
results = {}
for _ in range(5):
    for color_model in ['RGB', 'LA']:
        for loss_metric in loss_provider.loss_functions:
            if loss_metric == 'Watson-vgg':
                continue
            key = loss_metric + ' ' + color_model
            if key not in results:
                results[key] = {}
                results[key]['runtime'] = []
                results[key]['memory'] = []
            loss_function = loss_provider.get_loss_function(loss_metric, color_model, image_size=(3,64,64))
            data = x if color_model == 'RGB' else x_la
            res =  runtime_test(data, loss_function, epochs=500)
            results[key]['runtime'].append(res['runtime'])
            results[key]['memory'].append(res['memory'])

pickle.dump(results, open(os.path.join(g_drive_path, 'runtime_results_repitition.pickle'), 'wb'))
    


# # Print results.

# In[12]:


for model in results:
    print('{}: runtime mean: {}s, max memory {}Mb'.format(model, np.mean(results[model]['runtime']), max(results[model]['memory'])))


# In[13]:


print(torch.cuda.get_device_name(device=None))

