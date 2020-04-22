"""
arranges cases of either watson_fft or 
"""
import pandas as pd
import shutil
import os
from pathlib import Path
from PIL import Image

DATASET_DIR = "../perceptual_sim_training/"
TARGET_DIR = "../perceptual_sim_training/dataset/compare/"


def make_figure(dataframe, fig_name):
    frame_color_human_choice = 0
    boarder_width = 2
    img_spacing = 8
    max_images = 8
    img_size = 64

    # make canvas
    canvas = Image.new(
        "RGB",
        ((img_size + img_spacing) * max_images, (img_size + img_spacing) * 3),
        color="white",
    )

    # copy images on
    for i in range(max_images):
        print(dataframe.head(2))
        break

    canvas.save(fig_name)


# read data
data = pd.read_csv("comparison.csv")

# add a new column, comparing score between watson_fft and deeploss_vgg
data.insert(
    len(data.columns), "Wat-Deep", data["Watson-fft_score"] - data["Deeploss-vgg_score"]
)

# select rows where one of the functions does overwhelmingly better
wat_better = data[data["Wat-Deep"] == 1.0]
deep_better = data[data["Wat-Deep"] == -1.0]

# combine datasets
filtered_data = wat_better.append(deep_better)

# shuffle data
filtered_data = filtered_data.sample(frac=1).reset_index(drop=True)

make_figure(filtered_data, "fig.png")
