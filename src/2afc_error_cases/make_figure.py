"""
arranges cases of either watson_fft or 
"""
import pandas as pd
import shutil
import os
from pathlib import Path
from PIL import Image

# visually interesting samples for the paper
PRESENTATION_SAMPLES = [35924, 36305, 30973, 31952, 20341, 35499, 31642, 31904]


def make_figure(dataframe, fig_name):
    def paste_img(img, col, row, human_preferred=False):
        spacer_h = img_spacing_h + img_size
        spacer_v = img_spacing_v + img_size
        frame = Image.new(
            "RGB",
            (img_size + boarder_width * 2, img_size + boarder_width * 2),
            color=frame_color_human_choice
            if human_preferred
            else frame_color_not_human_choice,
        )
        canvas.paste(
            frame,
            (
                spacer_h * row + img_spacing_h // 2 - boarder_width,
                spacer_v * col + img_spacing_v // 2 - boarder_width,
            ),
        )
        canvas.paste(
            img,
            (spacer_h * row + img_spacing_h // 2, spacer_v * col + img_spacing_v // 2),
        )

    frame_color_human_choice = "#ee8080"
    frame_color_not_human_choice = "#cbcbcb"
    boarder_width = 8
    img_spacing_h = 40
    img_spacing_v = 28
    max_images = 8
    img_size = 256

    # make canvas
    canvas = Image.new(
        "RGB",
        ((img_size + img_spacing_h) * max_images, (img_size + img_spacing_v) * 3),
        color="#ffffff",
    )

    # copy images on
    for i in range(max_images):
        row = dataframe.loc[i]
        # reference img
        ref_fig = Image.open(row["ref_path"])
        paste_img(ref_fig, 0, i)

        human_pick = int(row["judge_pred"])
        watson_pick = int(row["Watson-fft_pred"])
        deeploss_pick = int(row["Deeploss-vgg_pred"])
        img_paths = [row["p0_path"], row["p1_path"]]

        # watson-picked img
        wat_fig = Image.open(img_paths[watson_pick])
        paste_img(wat_fig, 1, i, watson_pick == human_pick)

        # deeploss-picked img
        deeploss_fig = Image.open(img_paths[deeploss_pick])
        paste_img(deeploss_fig, 2, i, deeploss_pick == human_pick)

    canvas.save(fig_name)


# read data
data = pd.read_csv("comparison.csv")

# add a new column, comparing score between watson_fft and deeploss_vgg
data.insert(
    len(data.columns), "Wat-Deep", data["Watson-fft_score"] - data["Deeploss-vgg_score"]
)

# presentation samples
presentation_data = (
    data.iloc[PRESENTATION_SAMPLES].sample(frac=1).reset_index(drop=True)
)
make_figure(presentation_data, "2AFC_error_cases_4_paper.png")


# draw random samples
# select rows where one of the functions does overwhelmingly better
wat_better = data[data["Wat-Deep"] == 1.0]
deep_better = data[data["Wat-Deep"] == -1.0]

# combine datasets
filtered_data = wat_better.append(deep_better)


for i in range(10):
    # shuffle data
    filtered_data = filtered_data.sample(frac=1, random_state=i).reset_index(drop=True)

    # make fig
    make_figure(filtered_data, f"2AFC_error_cases_random_{i}.png")
