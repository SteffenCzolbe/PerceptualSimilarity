# A fast and memory efficient perceptual similarity metric for deep neural networks based on ideas from psychophysical research

Steffen Czolbe, 2019

<img src='./img/titleimage.png' width=500>

This repository contains the similarity metrics designed and tested in the thesis, as well as implementations of experiments. Implementation in the deep-learning framework PyTorch. Code intended to be reuseable is in neatly organized python files. Code not intended to be reuseable is implemented in jupyter notebooks.

## Requirements
The project is implemented in python. Requirements can be installed via
```
pip3 install requirements.txt
```
It is recommended to use a virtual environment. This will keep changes to your local python installation contained. 

## Using the similarity metrics

We implemented a ``LossProvider`` to easily make all pre-trained similarity metrics accessible. The provided code snipped imports the pre-trained Watson-fft metric for datasets with 3 channels. 
```python
from loss.loss_provider import LossProvider

provider = LossProvider()
loss_function = provider.get_loss_function('Watson-fft', 'RGB', reduction='sum')
```

You can further specify the returned loss function by changing the parameters:

* The first parameter defines the loss metric. Implemented metrics are ``'L1', 'L2', 'SSIM', 'Watson-dct', 'Watson-fft', 'Watson-vgg', 'Deeploss-vgg'``. 
* The second parameter defines the color representation. Default is the three-channel ``'RGB'`` model. Mono-channel Grey-scale representation can be used py passing ``'LA'``. 
* Keyword argument ``reduction``, with behaviour according to PyTorch guideline. Default value is ``reduction='sum'``. All metrics further support option ``reduction='none'``.
* Keyword argument ``deterministic``. Determines the shifting behaviour of metrics ``'Watson-dct'`` and ``'Watson-fft'``. The shifts make the metric non-deterministic, but lead to faster convergence and better results. Though in some cases we might prefer a deterministic behaviour. Default ``deterministic=False``.

## Datasets
Download scripts for the MNIST and celebA datasets are supplied in the ``src/datasets`` directory. 

## Re-training VAEs
The VAEs from the Thesis can be retrained with code from directories ``src/vae/mnist`` and ``src/vae/celebA``. Each directory contains 4 Files:
* the implementation of the model in `celebA_vae.py` or `mnist_vae.py` 
* a notebook  `train.ipynb`, contained code to train the model
* a directory `results`, in which models and sample images are saved during training
* a notebook `evaluation.ipynb`, containg code to generated the comparison images listed in the report.
* a directory `comparison`, in which comparison images are saved


## Agreement with human similarity judgements
The tuning and evaluation on the 2AFC dataset is located in directory `src/perceptual-sim-training`. The code is adapted from Zhang et. al. [2018], and includes adjustments to train and evaluate our own metrics. The directory `src/perceptual-sim-training/scripts` contains bash-scripts to download the dataset, train all metrics on color and grey-scale data, and evaluate each metric.

## Runtime Experiment
The runtime experiment is located in directory `src/runtime`. File `runtime_experiment.ipynb` contains the code to conduct the experiment, and file `runtime-barplot.ipynb` visualizes the result as a barplot.

## Convergence Experiment
The runtime experiment is located in directory `src/convergence`.

