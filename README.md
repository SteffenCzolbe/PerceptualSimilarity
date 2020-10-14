# A Loss Function for Generative Neural Networks Based on Watson’s Perceptual Model

## Steffen Czolbe, Oswin Krause, Igemar Cox, Christian Igel
## NeurIPS 2020

[[Arxiv Paper]](https://arxiv.org/abs/2006.15057) [Video]

<img src='./img/titleimage.png' width=500>

This repository contains the similarity metrics designed and evaluated in the [paper](https://arxiv.org/abs/2006.15057), and instructions and code to re-run the experiments. Implementation in the deep-learning framework PyTorch. Code supplied in Python 3 files and Jupyter Notebooks.

## Dependencies
The project is implemented in python. Dependencies can be installed via
```bash
$ pip3 install -r requirements.txt
```
Alternative: It is recommended to use a virtual environment. This will keep changes to your local python installation contained. First, we set up a virtual environment called ``venv-perceptual-sim``:
```bash
sudo apt-get install python3-pip
python3 -m pip install virtualenv
python3 -m virtualenv venv-perceptual-sim
```
Next, we activate it and install our dependencies
```bash
source venv-perceptual-sim/bin/activate
pip3 install -r requirements.txt 
```
If you want to use this kernel within jupyter notebooks, you also need to register it as an ipython kernel.
```bash
ipython kernel install --user --name=venv-perceptual-sim
```

# Use the similarity metrics

The presented similarity metrics can be included in your projects by importing the ``LossProvider``. It makes all pre-trained similarity metrics accessible. The example below shows how to build the ``Watson-DFT`` metric, and loads the weights tuned on the 2AFC dataset. 

```python
from loss.loss_provider import LossProvider

provider = LossProvider()
loss_function = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
```

Parameters:

* The first parameter defines the loss metric. Implemented metrics are ``'L1', 'L2', 'SSIM', 'Adaptive', 'Watson-DCT', 'Watson-DFT', 'Deeploss-VGG', 'Deeploss-Squeeze'``. 
* Keyword argument ``colorspace`` defines the color representation and dimensionality of the input. Default is the three-channel ``'RGB'`` model. Mono-channel greyscale representation can be used py passing ``'LA'``. 
* Keyword argument ``pretrained``. If `True` (default), the weights pre-trained on the 2AFC task are loaded.
* Keyword argument ``reduction``, with behaviour according to PyTorch guideline. Default value is ``reduction='sum'``. All metrics further support option ``reduction='none'``.
* Keyword argument ``deterministic``. Determines the shifting behaviour of metrics ``'Watson-DCT'`` and ``'Watson-DFT'``. The shifts make the metric non-deterministic, but lead to faster convergence and better results. Though in some cases we might prefer a deterministic behaviour. Default ``deterministic=False``.
* Keyword argument ``image_size``. Only required for `'Adaptive'`-Loss, as the implementation provided by the authors requires the input size. Example: `image_size=(3, 64, 64)`.

# Experiments
<span style="color:red">WARNING:</span> This part of the codebase is extremely unorganized. It was created as part of a master thesis, whithout much experience on how to write or maintain code for research. There is no neat `run_all_experiments_and_make_me_beacon.sh` script here. We provide the code as is, as we believe it is still helpfull for those willing and determined enough to work with it. You have been warned.


The following instructions allow you to re-run all experiments presented in the paper.


## Download Data
We use three datasets: MNIST, celebA and 2AFC. Execute the following commands to download all three datasets:
```bash
cd ./src/
PSHOME=$(pwd)
cd $PSHOME/datasets/MNIST
python3 mnist_download.py
cd $PSHOME/datasets/celebA
python3 celebA_download.py
cd $PSHOME/perceptual_sim_training
./scripts/download_dataset.sh
```

## Tune on the 2AFC Set
The tuning and evaluation on the 2AFC dataset is located in directory `src/perceptual-sim-training`. The code is adapted from Zhang et. al. [2018], and includes adjustments to train and evaluate our Modified Watson Metrics. 
### Train
To train the metrics, we execute the provided training script.
```bash
cd $PSHOME/perceptual_sim_training
./scripts/train.sh
```

### Evaluate
The evaluation of the metrics performance on the validation section of the 2AFC dataset is performed in jupyter notebooks. We start jupyter:
```bash
jupyter notebook
```
Wait for jupyter to open in your browser. Navigate to ``src/perceptual-sim-training/eval.ipynb``. Open the notebook, and select your virtual environment, and execute all cells. The 2AFC scores are written to a file.

To generate the plots from the paper, navigate to the notebook ``src/perceptual-sim-training/eval-plot.ipynb`` and execute it.

## Transition weights to the LossProvider
To transition loss functions tuned on the 2AFC dataset to the loss provider, we open the jupyter notebook ``src/perceptual-sim-training/transition_weights.ipynb`` and execute it. The lates model checkpoints from the 2AFC experiment are extracted, renamed and saved into the he 2AFC dataset to the loss provider, we open the jupyter notebook ``src/loss/weights/`` directory.

## Train VAEs
The VAEs presented in the paper can be retrained with code from directories ``src/vae/mnist`` and ``src/vae/celebA``. Each directory contains 5 Files:
* the implementation of the model in `celebA_vae.py` or `mnist_vae.py` 
* a notebook  `train.ipynb`, containing code to train the model. Loss function and Hyperparameters are defined in the last cell. During training, model checkpoints and samples are saved in the `results` directory.
* a directory `results`, in which models and sample images are saved during training
* a notebook `evaluation.ipynb`, containg code to generate the comparison images shown in the paper.
* a directory `comparison`, in which comparison images are saved


## Measure Resource Requirements
The resource experiment is located in `src/runtime/runtime_experiment.py`. It will run all loss functions multiple times, and print out the averaged measurments at the end. Results are also saved as as a pickle file.


