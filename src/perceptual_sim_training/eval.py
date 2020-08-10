#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from models import dist_model as dm
from data import data_loader as dl
import argparse
from IPython import embed
import torch
import torch.nn as nn
from collections import defaultdict
from data.twoafc_dataset import TwoAFCDataset
import matplotlib.pyplot as plt
import pickle
import os


# In[12]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
is_cuda = torch.cuda.is_available()

datasets = ['val/traditional','val/cnn','val/superres','val/deblur','val/color','val/frameinterp']
dataset_mode = '2afc' # 2afc or jnd

models = []


# # Grayscale models

# In[7]:

"""
# traditional Baselines

m = dm.DistModel()
m.initialize(model='L1',colorspace='Gray',use_gpu=is_cuda)
m.model_name = 'L1 [LA]'
models.append(m)

m = dm.DistModel()
m.initialize(model='L2',colorspace='Gray',use_gpu=is_cuda)
m.model_name = 'L2 [LA]'
models.append(m)

m = dm.DistModel()
m.initialize(model='ssim',colorspace='Gray',use_gpu=is_cuda)
m.model_name = 'SSIM [LA]'
models.append(m)


# In[ ]:


# watson models

m = dm.DistModel()
path=os.path.join('./checkpoints/', 'gray_watson_dct_trial0', 'latest_net_.pth')
m.initialize(model='watson', net='dct', colorspace='Gray',use_gpu=is_cuda)
m.net.load_state_dict(torch.load(path))
m.model_name = 'Watson-dct [LA]'
models.append(m)

m = dm.DistModel()
path=os.path.join('./checkpoints/', 'gray_watson_fft_trial0', 'latest_net_.pth')
m.initialize(model='watson', net='fft', colorspace='Gray',use_gpu=is_cuda)
m.net.load_state_dict(torch.load(path))
m.model_name = 'Watson-fft [LA]'
models.append(m)


# In[ ]:


# Deeploss competitor: lin tuned vgg
m = dm.DistModel()
path=os.path.join('./checkpoints/', 'gray_pnet_lin_vgg_trial0', 'latest_net_.pth')
m.initialize(model='net-lin', net='vgg', model_path=path, colorspace='Gray',use_gpu=is_cuda, batch_size=64)
m.model_name = 'Deeploss-vgg [LA]'
m.batch_size = m.batch_size // 4
models.append(m)


# In[3]:


# Deeploss competitor 2: lin tuned squeezenet
m = dm.DistModel()
path=os.path.join('./checkpoints/', 'gray_pnet_lin_squeeze_trial0', 'latest_net_.pth')
m.initialize(model='net-lin', net='squeeze', model_path=path, colorspace='Gray',use_gpu=is_cuda, batch_size=64)
m.model_name = 'Deeploss-squeeze [LA]'
m.batch_size = m.batch_size // 4
models.append(m)


# # color models

# In[ ]:


# traditional Baselines
m = dm.DistModel()
m.initialize(model='L1',colorspace='RGB',use_gpu=is_cuda)
m.model_name = 'L1'
models.append(m)

m = dm.DistModel()
m.initialize(model='L2',colorspace='RGB',use_gpu=is_cuda)
m.model_name = 'L2'
models.append(m)

m = dm.DistModel()
m.initialize(model='ssim',colorspace='RGB',use_gpu=is_cuda)
m.model_name = 'SSIM'
models.append(m)


# In[ ]:


# watson models

m = dm.DistModel()
path=os.path.join('./checkpoints/', 'rgb_watson_dct_trial0', 'latest_net_.pth')
m.initialize(model='watson', net='dct', colorspace='RGB',use_gpu=is_cuda)
m.net.load_state_dict(torch.load(path))
m.model_name = 'Watson-dct'
models.append(m)

m = dm.DistModel()
path=os.path.join('./checkpoints/', 'rgb_watson_fft_trial0', 'latest_net_.pth')
m.initialize(model='watson', net='fft', colorspace='RGB',use_gpu=is_cuda)
m.net.load_state_dict(torch.load(path))
m.model_name = 'Watson-fft'
models.append(m)


# In[ ]:


# Deeploss competitor: lin tuned vgg
m = dm.DistModel()
path=os.path.join('./checkpoints/', 'rgb_pnet_lin_vgg_trial0', 'latest_net_.pth')
m.initialize(model='net-lin', net='vgg', model_path=path, colorspace='RGB',use_gpu=is_cuda, batch_size=64)
m.model_name = 'Deeploss-vgg'
m.batch_size = m.batch_size // 4
models.append(m)


# In[3]:


# Deeploss competitor 2: lin tuned squeeze
m = dm.DistModel()
path=os.path.join('./checkpoints/', 'rgb_pnet_lin_squeeze_trial0', 'latest_net_.pth')
m.initialize(model='net-lin', net='squeeze', model_path=path, colorspace='RGB',use_gpu=is_cuda, batch_size=64)
m.model_name = 'Deeploss-squeeze'
m.batch_size = m.batch_size // 4
models.append(m)
"""

# # Adaptive loss, as per reviewer request

# In[13]:


m = dm.DistModel()
path=os.path.join('./checkpoints/', 'gray_adaptive_trial0', 'latest_net_.pth')
m.initialize(model='adaptive', colorspace='Gray',use_gpu=is_cuda)
m.net.load_state_dict(torch.load(path, map_location=device))
m.model_name = 'Adaptive [LA]'
models.append(m)

m = dm.DistModel()
path=os.path.join('./checkpoints/', 'rgb_adaptive_trial0', 'latest_net_.pth')
m.initialize(model='adaptive', colorspace='RGB',use_gpu=is_cuda)
m.net.load_state_dict(torch.load(path, map_location=device))
m.model_name = 'Adaptive'
models.append(m)


# # Evaluate

# In[9]:


def ddict(d):
    return defaultdict(lambda: defaultdict(lambda: {}), d)

def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)

def dict2ddict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict2ddict(v)
    return ddict(d)


# In[10]:


# initialize data loader
def eval_loss_metric(model, resultdict=None):
    if resultdict is None:
        resultdict = defaultdict(lambda: defaultdict(lambda: {}))
    
    for dataset in datasets:
        
        data_loader = dl.CreateDataLoader(dataset, dataset_mode=dataset_mode, batch_size=model.batch_size)

        # evaluate model on data
        if(dataset_mode=='2afc'):
            (score, results_verbose) = dm.score_2afc_dataset(data_loader, model.forward)
            resultdict[model.model_name][dataset]['score'] = results_verbose['scores'].mean()
            resultdict[model.model_name][dataset]['std'] = results_verbose['scores'].std()
            
            human_judgements = results_verbose['gts']
            human_scores = human_judgements**2 + (1 - human_judgements)**2
            resultdict['Human'][dataset]['score'] = human_scores.mean()
            resultdict['Human'][dataset]['std'] = human_scores.std()
            
            
        elif(dataset_mode=='jnd'):
            raise Exception('not implemented / validated')
            (score, results_verbose) = dm.score_jnd_dataset(data_loader, model.forward)


        # print results
        print(' Model [%s]  Dataset [%s]: %.2f'%(model.model_name, dataset, 100.*score))
    return resultdict


# In[ ]:


res = dict2ddict(pickle.load(open("eval_results.p", "rb")))
#res = defaultdict(lambda: defaultdict(lambda: {}))

for i, model in enumerate(models):
    print('eval model {} / {}: "{}"'.format(i+1, len(models), model.model_name))
    eval_loss_metric(model, res)

# save results
pickle.dump(ddict2dict(res), open("eval_results.p", "wb"))

