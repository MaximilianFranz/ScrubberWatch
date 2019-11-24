#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import click
import os

from tqdm import tqdm_notebook as tqdm
from torchvision.models import *
from fastai.vision import *
from fastai.vision.models import *
from fastai.vision.learner import model_meta
from fastai.callbacks import *

import sys


@click.command()
@click.argument("image-path")
def main(image_path):
    train = pd.read_csv("train/train.csv")

    image = open_image(image_path)
    CLASSES = {"no smoke": 1, "smoke": 2}
    CLASSES_INV = {v: k for k, v in CLASSES.items()}

    learn = load_learner(path="smoke", file="export.pkl")
    pred_category, _, _ = learn.predict(image)  # [3, 256, 256]
    pred_category = int(pred_category.data.numpy())

    print(
        "Predicted category : {}, {}".format(pred_category, CLASSES_INV[pred_category])
    )


if __name__ == "__main__":
    main()
