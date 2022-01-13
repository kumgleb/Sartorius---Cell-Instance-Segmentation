import toml
import wandb
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn, optim
from src.training import train_model
from src.data_utils import CellDataset
from src.model import ResUNet
from src.losses import FocalLoss, ExponentialLogarithmicLoss, MixedLoss

import albumentations as albu
from albumentations.pytorch import ToTensorV2
from src.transfroms import Normalize, RandomCrop, HardCutout

from src.utils import set_seed


torch.cuda.empty_cache()

wandb.init(project="sartorius")
cfg = toml.load("./config/cfg.toml")
wandb.config = cfg
set_seed(cfg["seed"])


train_transforms = albu.Compose(
    [
        albu.RandomScale((0.7, 1.3), p=0.2),
        albu.Rotate(limit=30, p=0.2),
        RandomCrop((224, 224)),
        albu.Flip(p=0.2),
        albu.RandomRotate90(p=0.2),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                ),
                albu.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50
                ),
            ],
            p=0.2,
        ),
        albu.Cutout(
            max_h_size=int(224 * 0.1), max_w_size=int(224 * 0.1), num_holes=5, p=0.05
        ),
        HardCutout((122, 122), p=0.02),
        Normalize(),
        ToTensorV2(),
    ]
)

val_transforms = albu.Compose([Normalize(), RandomCrop((224, 224)), ToTensorV2()])


IMG_FOLDER = "./data/train"
train_df = pd.read_csv("./data/train.csv")
train_ids = np.load("./data/data_split/train_idxs.npy", allow_pickle=True)
val_ids = np.load("./data/data_split/val_idxs.npy", allow_pickle=True)

train_dataset = CellDataset(IMG_FOLDER, train_df, train_ids, train_transforms)
val_dataset = CellDataset(IMG_FOLDER, train_df, val_ids, val_transforms)

dataloader_train = DataLoader(
    train_dataset, batch_size=cfg["train"]["bs"], shuffle=True
)
dataloader_val = DataLoader(val_dataset, batch_size=cfg["train"]["bs"], shuffle=False)

print("Train size: ", len(dataloader_train))
print("Val size: ", len(dataloader_val))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

resunet = ResUNet(cfg["model"]).to(device)

optimizer = optim.Adam(resunet.parameters(), lr=cfg["train"]["lr"])
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=1000, gamma=0.9, verbose=True
)

criterion = MixedLoss(1, 1, 0.5)


if __name__ == "__main__":
    train_model(
        resunet,
        dataloader_train,
        dataloader_val,
        device,
        optimizer,
        criterion,
        scheduler,
        cfg,
    )
