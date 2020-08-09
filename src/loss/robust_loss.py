from robust_loss_pytorch.robust_loss_pytorch import AdaptiveImageLossFunction
import torch
import torch.nn as nn

class RobustLoss(nn.Module):
    """
    Robust loss from the papaer

    A General and Adaptive Robust Loss Function

    by Jonathan T. Barron CVPR, 2019

    wrapper for his original code, to conform with our interface
    we use his loss with DCT decomposition and YUV colorspace
    """
    def __init__(self, image_size, use_gpu=True, trainable=False, reduction='mean'):
        """
        Parameters:
        image_size: the size of the image, CxHxW
        trainable: bool, if True parameters of the loss are trained and dropout is enabled.
        reduction: 'sum' or 'none', determines return format
        """
        super().__init__()
        self.reduction = reduction
        
        # module to perform feature extraction
        self.add_module('robust_loss', AdaptiveImageLossFunction(image_size=image_size,
               float_dtype=torch.float32,
               device='cuda:0' if use_gpu else 'cpu',
               color_space='YUV',
               representation='DCT',
               use_students_t=False,
               alpha_lo=0.001,
               alpha_hi=1.999,
               scale_lo=1e-5,
               scale_init=1.0))
        
    def forward(self, input, target):
        diff = input - target
        distance = self.robust_loss.lossfun(diff)

        # reduce
        if self.reduction == 'sum':
            distance = torch.sum(distance)
        elif self.reduction == 'mean':
            distance = torch.mean(distance)
        return distance

if __name__ == '__main__':
    import torch
    a = torch.rand(2,3,128,128)
    b = torch.rand(2,3,128,128)

    l = RobustLoss(a.shape[1:], use_gpu=False)

    loss = l(a, b)
    print(loss)
    loss = l(a, a)
    print(loss)
    
