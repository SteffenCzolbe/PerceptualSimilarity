import torch
import numpy as np
from data import data_loader as dl
from models import dist_model as dm
import pandas as pd
import os
import sys
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.abspath("../loss"))
from loss.loss_provider import LossProvider


if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else "cpu"

    # load dataset
    datasets = [
        "val/traditional",
        "val/cnn",
        "val/superres",
        "val/deblur",
        "val/color",
        "val/frameinterp",
    ]
    dataset_mode = "2afc"
    data_loader = dl.CreateDataLoader(
        datasets, dataset_mode=dataset_mode, batch_size=1
    ).load_data()

    # load functions to evaluate
    lp = LossProvider()
    metrics = {}
    metrics["Watson-fft"] = lp.get_loss_function("Watson-fft", reduction="sum").to(device)
    metrics["L2"] = lp.get_loss_function("L2", reduction="sum").to(device)
    metrics["SSIM"] = lp.get_loss_function("SSIM", reduction="sum").to(device)
    metrics["Deeploss-vgg"] = lp.get_loss_function("Deeploss-vgg", reduction="sum").to(device)

    # set-up output file
    out_data = defaultdict(lambda: [])

    # score
    for data in tqdm(data_loader):
        out_data["ref_path"] += data["ref_path"]
        out_data["p0_path"] += data["p0_path"]
        out_data["p1_path"] += data["p1_path"]

        gt = data["judge"].flatten().to(device)
        out_data["judge_exact"] += gt.tolist()
        out_data["judge_pred"] += gt.round().tolist()

        ref = data["ref"].to(device)
        p0 = data["p0"].to(device)
        p1 = data["p1"].to(device)

        for metric, fun in metrics.items():
            d0 = fun(p0, ref)
            d1 = fun(p1, ref)
            out_data[f"{metric}_d0"] += [d0.item()]
            out_data[f"{metric}_d1"] += [d1.item()]
            out_data[f"{metric}_score"] += [((
                (d0 < d1) * (1.0 - gt) + (d1 < d0) * gt + (d1 == d0) * 0.5
            )).item()]
            out_data[f"{metric}_pred"] += [((d0 > d1) * 1.0).item()]

    # write output data
    df = pd.DataFrame(out_data)
    df.to_csv("comparison.csv")
