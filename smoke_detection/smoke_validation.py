#!/usr/bin/env python

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import os

from tqdm import tqdm_notebook as tqdm
from torchvision.models import *

from fastai.vision import *
from fastai.vision.models import *
from fastai.vision.learner import model_meta
from fastai.callbacks import *
import sys

train = pd.read_csv("train/train_smoke.csv")

LABEL_DICT = {"no_smoke": 1, "smoke": 2}

for k, v in LABEL_DICT.items():
    train.category[train.category == v] = k
train.to_csv("train/train_smoke2.csv", index=False)

path = pathlib.Path("train")
tfms = get_transforms(
    do_flip=True,
    max_rotate=20.0,
    p_affine=0.75,
    max_lighting=0.5,
    max_warp=0.3,
    p_lighting=0.75,
)
np.random.seed(31)
data = ImageDataBunch.from_csv(
    path,
    folder="images",
    csv_labels="train_smoke2.csv",
    valid_pct=0.15,
    test="test",
    ds_tfms=tfms,
    size=256,
    bs=32,
)
data.show_batch(rows=3, figsize=(5, 5))

fbetaW = FBeta(beta=1, average="weighted")
# learn = load_learner(path="smoke", file="export.pkl")
learn = create_cnn(
    data,
    models.resnet101,
    metrics=[accuracy, fbetaW],
    model_dir="./checkpoints",
    opt_func=optim.SGD,
    path="smoke",
)
learn = learn.load("resnet-101")

interp = ClassificationInterpretation.from_learner(learn)


# import ipdb

# ipdb.set_trace()
NB_RC_PARAMS = {
    "figure.figsize": [5, 4],
    "figure.dpi": 220,
    "figure.autolayout": True,
    "font.size": 16,
}
with plt.rc_context(NB_RC_PARAMS):
    fig = interp.plot_confusion_matrix(figsize=(5, 4), return_fig=True, title=None)
    ax = plt.gca()
    ax.set_xticklabels(["no_smoke", "smoke"], rotation=0)
    fig.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(learn.path, "confusion.png"))

