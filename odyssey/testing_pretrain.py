import numpy as np
import pandas as pd
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from odyssey.models.ehr_mamba.model import MambaPretrain

from odyssey.utils.utils import seed_everything

data_path = "./odyssey/P12data/"

data_train = pd.read_pickle(data_path + "train_df.pkl")
data_test = pd.read_pickle(data_path + "test_df.pkl")
data_val = pd.read_pickle(data_path + "validation_df.pkl")

seed_everything(42)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("medium")

train_loader = DataLoader(
    data_train,
    batch_size=10,
    num_workers=1,
    persistent_workers=False,
    shuffle=True,
    pin_memory=True,
)
val_loader = DataLoader(
    data_val,
    batch_size=10,
    num_workers=1,
    persistent_workers=False,
    shuffle=False,
    pin_memory=True,
)

print("building model")
model = MambaPretrain(
    vocab_size=37)