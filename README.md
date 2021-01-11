# A Loss Function for Generative Neural Networks Based on Watson’s Perceptual Model

Steffen Czolbe, Oswin Krause, Igemar Cox, Christian Igel - NeurIPS 2020

[[Paper]](https://arxiv.org/abs/2006.15057) [[Video]](https://youtu.be/qPmHQbR4DeI) [[Poster]](img/WatsonPoster.pdf)

[![Video](https://img.youtube.com/vi/qPmHQbR4DeI/hqdefault.jpg)](https://youtu.be/qPmHQbR4DeI)

This repository contains the similarity metrics designed and evaluated in the [paper](https://arxiv.org/abs/2006.15057), and instructions and code to re-run the experiments. Implementation in the deep-learning framework PyTorch. Code supplied in Python 3 files and Jupyter Notebooks.

> **Note**: This is the post-publication updated version of the repository. It contains the following changes:
>
> - Fixed an issue leading to inconsistent data-normalization across experiments. All experiments now take data normalized to the 0..1 range.
> - Re-tuned hyperparameters and re-generated all figures from the paper. We observed overall similar results. See a side-by-side comparison [here](https://docs.google.com/presentation/d/1Rc1N09-ZaP03TmVljAN4IQVIvnIJ4f6UgEc7RGccECs/edit?usp=sharing).
> - Added multiple perviously not included dependencies.
> - Added multiple shell-scripts to reproduce all experiments more easily. Jupyter notebook is no longer required to reproduce the paper \o/

# Use the similarity metrics

The presented similarity metrics can be included in your projects by importing the `LossProvider`. It makes all pre-trained similarity metrics accessible. The example below shows how to build the `Watson-DFT` metric, and loads the weights tuned on the 2AFC dataset. The input for all loss functions is expected to be normalized to a 0..1 interval.

```python
from loss.loss_provider import LossProvider

provider = LossProvider()
loss_function = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')

import torch
img0 = torch.zeros(1,3,64,64)
img1 = torch.zeros(1,3,64,64)
loss = loss_function(img0, img1)
```

Parameters:

- The first parameter defines the loss metric. Implemented metrics are `'L1', 'L2', 'SSIM', 'Adaptive', 'Watson-DCT', 'Watson-DFT', 'Deeploss-VGG', 'Deeploss-Squeeze'`.
- Keyword argument `colorspace` defines the color representation and dimensionality of the input. Default is the three-channel `'RGB'` model. Mono-channel greyscale representation can be used py passing `'LA'`.
- Keyword argument `pretrained`. If `True` (default), the weights pre-trained on the 2AFC task are loaded.
- Keyword argument `reduction`, with behaviour according to PyTorch guideline. Default value is `reduction='sum'`. All metrics further support option `reduction='none'`.
- Keyword argument `deterministic`. Determines the shifting behaviour of metrics `'Watson-DCT'` and `'Watson-DFT'`. The shifts make the metric non-deterministic, but lead to faster convergence and better results. Though in some cases we might prefer a deterministic behaviour. Default `deterministic=False`.
- Keyword argument `image_size`. Only required for `'Adaptive'`-Loss, as the implementation provided by the authors requires the input size. Example: `image_size=(3, 64, 64)`.

# Experiments

> **Warning:** This part of the codebase is unorganized. It was created as part of a master thesis, whithout much experience on how to write or maintain code for research. We have since made some progress to improve ease of use, but there is no neat `run_all_experiments_and_make_me_beacon.sh` script here. We provide the code as is, as we believe it is still helpfull for those willing and determined enough to work with it. You have been warned.

## Dependencies

The project is implemented in python. Dependencies can be installed via

```bash
$ pip3 install -r requirements.txt
```

Alternative: It is recommended to use a virtual environment. This will keep changes to your local python installation contained. First, we set up a virtual environment called `venv-perceptual-sim`:

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

Part of the codebase is implemented in jupyter notebooks (sorry). The provided scripts convert these automatically to python files, and executes those python files instead. Interaction with the notebooks is only required if you want to perform changes to the codebase.

## Download Data

We use three datasets: MNIST, celebA and 2AFC. Execute the following commands to download all three datasets:

Note: As of Dec-2020 the google-drive download of the celebA dataset is experiencing some issues. If you are having trouble with the script, we recomment manually downloading the file `img_align_celeba.zip` from [here](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing), placing it in the `$PSHOME/datasets/celebA` directory and renaming it to `celebA.zip`. Then the `celebA_download` script can betweaked by commenting out the downloading in line 55. The unpacking and pre-processing steps from this script still need to be executed.

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

### Evaluate and Generate Figures

The evaluation (Test) of the metrics performance on the validation section of the 2AFC dataset, and the Bar-plot from th paper is performed by the script

```bash
cd $PSHOME/perceptual_sim_training
./scripts/eval_and_plot.sh
```

The plots will be generated into `src/perceptual_sim_training/plots/`.

## Transition weights to the LossProvider

The loss-function weights have to be manually transitioned to the `LossProvider`, which will be used for the future experiments. This is done by calling the script

```bash
cd $PSHOME/perceptual_sim_training
./scripts/transition_weights.sh
```

The lates model checkpoints from the 2AFC experiment are extracted, renamed and saved into the he 2AFC dataset to the loss provider in the `src/loss/weights/` directory.

## Train VAEs

The VAEs presented in the paper can be retrained with code from directories `src/vae/mnist` and `src/vae/celebA`. Each directory contains 5 Files:

- the implementation of the model in `celebA_vae.py` or `mnist_vae.py`
- a file `train.py`, containing code to train the model. Loss function and Hyperparameters are defined at the end of the file. During training, model checkpoints and samples are saved in the `results` directory.
- a directory `results`, in which models and sample images are saved during training
- a notebook `evaluation.ipynb`, containg code to generate the comparison images shown in the paper.
- a directory `comparison`, in which comparison images are saved

## Measure Resource Requirements

The resource experiment is located in `src/runtime/runtime_experiment.py`. It will run all loss functions multiple times, and print out the averaged measurments at the end. Results are also saved as as a pickle file.

# Thanks

We thank Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang for their Two Alternative Forced Choice (2AFC) dataset, and implementation of the perceptual tuning code. We adapted large parts of their [code](https://github.com/richzhang/PerceptualSimilarity) for this project.

We also thank Jonathan T. Barron for providing the implementation f his [General and Adaptive Robust Loss Function](https://github.com/jonbarron/robust_loss_pytorch), which we incuded in this project.
