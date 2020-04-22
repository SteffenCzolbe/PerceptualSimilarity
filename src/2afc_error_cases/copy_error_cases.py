"""
splits the 2AFC dataset based on scores from the eval_distances run
"""
import pandas as pd
import shutil
import os
from pathlib import Path

DATASET_DIR = "../perceptual_sim_training/"
TARGET_DIR = "../perceptual_sim_training/dataset/compare/"

# read data
data = pd.read_csv("comparison.csv")

# add a new column, comparing score between watson_fft and deeploss_vgg
data.insert(
    len(data.columns), "Wat-Deep", data["Watson-fft_score"] - data["Deeploss-vgg_score"]
)

# select rows where Watson-fft does overwhelmingly better
wat_better = data[data["Wat-Deep"] == 1.0]

# copy those files to separate directory
Path(os.path.join(TARGET_DIR, "Watson-fft_better")).mkdir(parents=True, exist_ok=True)
for ref_path, p0_path, p1_path, gt in zip(
    wat_better["ref_path"],
    wat_better["p0_path"],
    wat_better["p1_path"],
    wat_better["judge_pred"],
):
    no = ref_path.split("/")[-1].split(".")[0]
    shutil.copyfile(
        os.path.join(DATASET_DIR, ref_path),
        os.path.join(TARGET_DIR, "Watson-fft_better", no + "ref" + ".png"),
    )
    identifier = "p0_" if gt == 0 else "p0"
    shutil.copyfile(
        os.path.join(DATASET_DIR, p0_path),
        os.path.join(TARGET_DIR, "Watson-fft_better", no + identifier + ".png"),
    )
    identifier = "p1_" if gt == 1 else "p1"
    shutil.copyfile(
        os.path.join(DATASET_DIR, p1_path),
        os.path.join(TARGET_DIR, "Watson-fft_better", no + identifier + ".png"),
    )

# look at images where deeploss does better
deep_better = data[data["Wat-Deep"] == -1.0]
Path(os.path.join(TARGET_DIR, "Deep-vgg_better")).mkdir(parents=True, exist_ok=True)
for ref_path, p0_path, p1_path, gt in zip(
    deep_better["ref_path"],
    deep_better["p0_path"],
    deep_better["p1_path"],
    deep_better["judge_pred"],
):
    no = ref_path.split("/")[-1].split(".")[0]
    shutil.copyfile(
        os.path.join(DATASET_DIR, ref_path),
        os.path.join(TARGET_DIR, "Deep-vgg_better", no + "ref" + ".png"),
    )
    identifier = "p0_" if gt == 0 else "p0"
    shutil.copyfile(
        os.path.join(DATASET_DIR, p0_path),
        os.path.join(TARGET_DIR, "Deep-vgg_better", no + identifier + ".png"),
    )
    identifier = "p1_" if gt == 1 else "p1"
    shutil.copyfile(
        os.path.join(DATASET_DIR, p1_path),
        os.path.join(TARGET_DIR, "Deep-vgg_better", no + identifier + ".png"),
    )
