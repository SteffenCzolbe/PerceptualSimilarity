# A fast and memory efficient perceptual similarity metric for deep neural networks based on ideas from psychophysical research

Steffen Czolbe, 2019

<img src='./img/titleimage.png' width=1000>

This repository contains the similarity metrics designed and tested in the thesis, as well as some of the experiments. It is implemented in PyTorch. Code intended to be reuseable is in separate files, code not intended to be reuseable is implemented in jupyter notebooks.

## Using the similarity metrics

We implemented a ``LossProvider`` to easily make all pre-trained similarity metrics accessible. To use the Watson-fft metric, simply
```python
from loss.loss_provider import LossProvider

provider = LossProvider()
loss_function = provider.get_loss_function('Watson-fft', 'RGB', reduction='sum')
```

You can further specify the characteristics of the loss function with parameters:

* The first parameter defined the loss metric. Implemented metrics are ``'L1', 'L2', 'SSIM', 'Watson-dct', 'Watson-fft', 'Watson-vgg', 'Deeploss-vgg'``. 
* The second parameter defines the color representation. Default is the three-channel ``'RGB'`` model. Mono-channel Grey-scale representation can be used py passing ``'LA'``. 
* Keyword argument ``reduction``, with behaviour according to PyTorch guideline. Default value is ``reduction='sum'``. All metrics further support option ``reduction='none'``.
* Keyword argument ``deterministic``. Determines the shifting behaviour of metrics ``'Watson-dct'`` and ``'Watson-fft'``. Default ``deterministic=False``.

## Training generative models with the metrics
The experiments from the Thesis are implemented in directory ``src/vae``. They provide an example of how to train a VAE with the supplied similarity metrics.

## Datasets
Download scripts for the MNIST and celebA datasets are supplied in the ``src/datasets`` directory.

## Re-runnng Experiments
TBA.

