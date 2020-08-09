
from __future__ import absolute_import

import sys
sys.path.append('..')
sys.path.append('.')
sys.path.append('../loss')
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from pdb import set_trace as st
from skimage import color
from IPython import embed
from . import pretrained_networks as pn
from watson import WatsonDistance
from watson_fft import WatsonDistanceFft
from watson_vgg import WatsonDistanceVgg
from robust_loss import RobustLoss as RobustLossFunction
from color_wrapper import ColorWrapper

# from PerceptualSimilarity.util import util
from util import util

# Off-the-shelf deep network
class PNet(nn.Module):
    '''Pre-trained network with all channels equally weighted by default'''
    def __init__(self, pnet_type='vgg', pnet_rand=False, use_gpu=True, colorspace='RGB'):
        super(PNet, self).__init__()

        self.use_gpu = use_gpu

        self.colorspace = colorspace

        self.pnet_type = pnet_type
        self.pnet_rand = pnet_rand

        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1))
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1))
        
        if(self.pnet_type in ['vgg','vgg16']):
            self.net = pn.vgg16(pretrained=not self.pnet_rand,requires_grad=False)
        elif(self.pnet_type=='alex'):
            self.net = pn.alexnet(pretrained=not self.pnet_rand,requires_grad=False)
        elif(self.pnet_type[:-2]=='resnet'):
            self.net = pn.resnet(pretrained=not self.pnet_rand,requires_grad=False, num=int(self.pnet_type[-2:]))
        elif(self.pnet_type=='squeeze'):
            self.net = pn.squeezenet(pretrained=not self.pnet_rand,requires_grad=False)

        self.L = self.net.N_slices

        if(use_gpu):
            self.net.cuda()
            self.shift = self.shift.cuda()
            self.scale = self.scale.cuda()

    def forward(self, in0, in1, retPerLayer=False):
        in0_sc = (in0 - self.shift.expand_as(in0))/self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0))/self.scale.expand_as(in0)

        if self.colorspace == 'Gray':
            in0_sc = util.tensor2tensorGrayscaleLazy(in0_sc)
            in1_sc = util.tensor2tensorGrayscaleLazy(in1_sc)

        outs0 = self.net.forward(in0_sc)
        outs1 = self.net.forward(in1_sc)

        if(retPerLayer):
            all_scores = []
        for (kk,out0) in enumerate(outs0):
            cur_score = (1.-util.cos_sim(outs0[kk],outs1[kk]))
            if(kk==0):
                val = 1.*cur_score
            else:
                # val = val + self.lambda_feat_layers[kk]*cur_score
                val = val + cur_score
            if(retPerLayer):
                all_scores+=[cur_score]

        if(retPerLayer):
            return (val, all_scores)
        else:
            return val

# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, use_gpu=True, spatial=False, version='0.1', colorspace='RGB'):
        super(PNetLin, self).__init__()

        self.use_gpu = use_gpu
        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.version = version
        self.colorspace = colorspace

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = pn.vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = pn.squeezenet
            self.chns = [64,128,256,384,384,512,512]

        if(self.pnet_tune):
            self.net = net_type(pretrained=not self.pnet_rand,requires_grad=True)
        else:
            self.net = [net_type(pretrained=not self.pnet_rand,requires_grad=False),]

        self.lin0 = NetLinLayer(self.chns[0],use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1],use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2],use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3],use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4],use_dropout=use_dropout)
        self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
        if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
            self.lin5 = NetLinLayer(self.chns[5],use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6],use_dropout=use_dropout)
            self.lins+=[self.lin5,self.lin6]

        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1,3,1,1))
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1,3,1,1))

        if(use_gpu):
            if(self.pnet_tune):
                self.net.cuda()
            else:
                self.net[0].cuda()
            self.shift = self.shift.cuda()
            self.scale = self.scale.cuda()
            self.lin0.cuda()
            self.lin1.cuda()
            self.lin2.cuda()
            self.lin3.cuda()
            self.lin4.cuda()
            if(self.pnet_type=='squeeze'):
                self.lin5.cuda()
                self.lin6.cuda()

    def forward(self, in0, in1):
        in0_sc = (in0 - self.shift.expand_as(in0))/self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0))/self.scale.expand_as(in0)

        if self.colorspace == 'Gray':
            in0_sc = util.tensor2tensorGrayscaleLazy(in0_sc)
            in1_sc = util.tensor2tensorGrayscaleLazy(in1_sc)

        if(self.version=='0.0'):
            # v0.0 - original release had a bug, where input was not scaled
            in0_input = in0
            in1_input = in1
        else:
            # v0.1
            in0_input = in0_sc
            in1_input = in1_sc

        if(self.pnet_tune):
            outs0 = self.net.forward(in0_input)
            outs1 = self.net.forward(in1_input)
        else:
            outs0 = self.net[0].forward(in0_input)
            outs1 = self.net[0].forward(in1_input)

        feats0 = {}
        feats1 = {}
        diffs = [0]*len(outs0)

        for (kk,out0) in enumerate(outs0):
            feats0[kk] = util.normalize_tensor(outs0[kk])  # norm NN outputs
            feats1[kk] = util.normalize_tensor(outs1[kk])  
            diffs[kk] = (feats0[kk]-feats1[kk])**2         # squared diff

        if self.spatial:
            lin_models = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if(self.pnet_type=='squeeze'):
                lin_models.extend([self.lin5, self.lin6])
            res = [lin_models[kk].model(diffs[kk]) for kk in range(len(diffs))]
            return res
			
        val = torch.mean(torch.mean(self.lin0.model(diffs[0]),dim=3),dim=2)  # sum means over H, W
        val = val + torch.mean(torch.mean(self.lin1.model(diffs[1]),dim=3),dim=2)
        val = val + torch.mean(torch.mean(self.lin2.model(diffs[2]),dim=3),dim=2)
        val = val + torch.mean(torch.mean(self.lin3.model(diffs[3]),dim=3),dim=2)
        val = val + torch.mean(torch.mean(self.lin4.model(diffs[4]),dim=3),dim=2)
        if(self.pnet_type=='squeeze'):
            val = val + torch.mean(torch.mean(self.lin5.model(diffs[5]),dim=3),dim=2)
            val = val + torch.mean(torch.mean(self.lin6.model(diffs[6]),dim=3),dim=2)

        val = val.view(val.size()[0],val.size()[1],1,1)

        return val

class Watson(nn.Module):
    def __init__(self, transform='dct', use_gpu=True, is_train=False, colorspace='RGB', blocksize=8):
        super().__init__()

        self.colorspace = colorspace
        self.transform = transform
        
        reduction = 'none'

        net_name =  'Watson' 
        net_name += '-fft' if transform == 'FFT' else ''
        net_name += '-vgg' if transform == 'Vgg' else ''
        net_name += '-tune' if is_train else ''
        self.model_name = net_name
        
        if transform == 'dct':
            if self.colorspace == 'Gray':
                self.add_module('net', WatsonDistance(trainable=is_train, reduction=reduction))
            elif self.colorspace == 'RGB':
                self.add_module('net', ColorWrapper(WatsonDistance, (), {'trainable': is_train, 'blocksize': blocksize, 'reduction': reduction}, trainable=is_train))
        elif transform == 'fft':
            if self.colorspace == 'Gray':
                self.add_module('net', WatsonDistanceFft(trainable=is_train, reduction=reduction))
            elif self.colorspace == 'RGB':
                self.add_module('net', ColorWrapper(WatsonDistanceFft, (), {'trainable': is_train, 'blocksize': blocksize, 'reduction': reduction}, trainable=is_train))
        elif transform == 'vgg':
            self.add_module('net', WatsonDistanceVgg(trainable=is_train, reduction=reduction))
        else:
            raise Exception('Transform "{}" not implemented'.format(transform))
        
        if use_gpu:
            self.net = self.net.to('cuda')
    
    def forward(self, input, target):
        # input to 0 .. 1 scale
        input = (input + 1) / 2
        target = (target + 1) / 2

        # grayscale transform
        if self.colorspace == 'Gray' and self.transform in ['dct', 'fft']:
            # 1 channel input
            input = util.tensor2tensorGrayscale(input)
            target = util.tensor2tensorGrayscale(target)
        elif self.colorspace == 'Gray' and self.transform in ['vgg']:
            # 3 channel input
            input = util.tensor2tensorGrayscaleLazy(input)
            target = util.tensor2tensorGrayscaleLazy(target)


        # loss
        return self.net(input, target)


class RobustLoss(nn.Module):
    def __init__(self, size=[3,64,64], use_gpu=True, colorspace='RGB'):
        super().__init__()
        self.size = size
        self.colorspace = colorspace
        self.loss = RobustLossFunction(size, use_gpu=use_gpu, reduction='none')
        self.model_name = 'Adaptive Robust Loss Function' 

    def forward(self, in0, in1):
        if(self.colorspace=='Gray'):
            in0 = util.tensor2tensorGrayscaleLazy(in0)
            in1 = util.tensor2tensorGrayscaleLazy(in1)
        loss = self.loss(in0, in1)
        return torch.mean(loss, dim=[1,2,3])

class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32,use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

class BCERankingLoss(nn.Module):
    def __init__(self, use_gpu=True, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.use_gpu = use_gpu
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()
        self.model = nn.Sequential(*[self.net])

        if(self.use_gpu):
            self.net.cuda()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        d0 = d0.view(50,1,1,1)
        d1 = d1.view(50,1,1,1)
        if(self.use_gpu):
            per = per.cuda()
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)

class WeightedSigmoidLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(WeightedSigmoidLoss, self).__init__()
        self.use_gpu = use_gpu
        w_tile = torch.tensor([1.])   # w = exp(w_tild) to ensure weight >0
        if use_gpu:
            w_tilde = w_tile.cuda()
        self.w_tild = nn.Parameter(w_tilde)  
        self.zero = torch.tensor(0.)
        self.loss = torch.nn.BCELoss()
        self.parameters = list(self.parameters())  # for this strange frameworks sake

        if(self.use_gpu):
            self.loss = self.loss.cuda()
            self.w_tild = self.w_tild.cuda()
            self.zero = self.zero.cuda()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2. # scale judge to 0..1
        d0 = d0.view(50)
        d1 = d1.view(50)
        if(self.use_gpu):
            per = per.cuda()

        norm_dist = torch.where((d0 + d1) > 0, (d0 - d1) / (d0 + d1), self.zero) # save divison by 0
        weighted_sig = torch.sigmoid(torch.exp(self.w_tild) * norm_dist)  # weighted sigmoid of d0, d1 dist
        assert (weighted_sig >= 0).all() and (weighted_sig <= 1).all(), print(weighted_sig, d0, d1)
        return self.loss(weighted_sig, per) # Bin cross entropy with human judge

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace=colorspace

class L2(FakeNet):
    def forward(self, in0, in1):

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Gray'):
            (N,C,X,Y) = in0.size()
            in0 = util.tensor2tensorGrayscale(in0)
            in1 = util.tensor2tensorGrayscale(in1)

            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Lab'):
            assert(in0.size()[0]==1) # currently only supports batchSize 1
            value = util.l2(util.tensor2np(util.tensor2tensorlab(in0.data,to_norm=False)), 
                util.tensor2np(util.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var


class L1(FakeNet):
    def forward(self, in0, in1):

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1).abs(),dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Gray'):
            (N,C,X,Y) = in0.size()
            in0 = util.tensor2tensorGrayscale(in0)
            in1 = util.tensor2tensorGrayscale(in1)

            value = torch.mean(torch.mean(torch.mean((in0-in1).abs(),dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        else: 
            raise Exception('colorspace not implemented')

        

class DSSIM(FakeNet):

    def forward(self, in0, in1):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = util.dssim(1.*util.tensor2im(in0.data), 1.*util.tensor2im(in1.data), range=255.).astype('float')
        elif self.colorspace=='Gray':
            in0 = util.tensor2tensorGrayscaleLazy(in0)
            in1 = util.tensor2tensorGrayscaleLazy(in1)
            value = util.dssim(1.*util.tensor2im(in0.data), 1.*util.tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = util.dssim(util.tensor2np(util.tensor2tensorlab(in0.data,to_norm=False)), 
                util.tensor2np(util.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
        ret_var = Variable( torch.Tensor((value,) ) )
        if(self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var


class Edge(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Gray', args=None):
        super(Edge, self).__init__()

        size = args.get('size', 3)
        var = args.get('var', 1)
        var_delta = args.get('var_delta', 1)

        self.use_gpu = use_gpu

        self.colorspace = colorspace

        self.add_module('loss', EdgePreservingLoss(size=size, var=var, var_delta=var_delta, reduction='none'))

        if(use_gpu):
            self.loss = self.loss.cuda()

    def forward(self, in0, in1, retPerLayer=False):

        if self.colorspace == 'Gray':
            in0 = util.tensor2tensorGrayscale(in0)
            in1 = util.tensor2tensorGrayscale(in1)
        else:
            raise Exception('Edge loss does not support colorspaces other than grayscale')

        val = self.loss.forward(in0, in1)

        return val


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network',net)
    print('Total number of parameters: %d' % num_params)
