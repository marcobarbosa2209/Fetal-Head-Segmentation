# loads HC18 images and masks for pytorch, with data augmentation for training
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMG_H, IMG_W = 352, 512

# return the image/mask transformations for training or for validation/test
def get_transforms(train: bool):
    if train:

        # random changes in brightness/contrast, small rotations, shifts and zooms, horizontal flips
        # so the model sees slightly different versions of the same scans each epoch
        # helps it generalise instead of memorising
        return A.Compose([
            A.Resize(IMG_H, IMG_W),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                               border_mode=cv2.BORDER_CONSTANT, p=0.7),
            A.RandomBrightnessContrast(p=0.5),
        ])

    # for validation test, only resize, without randomness, so results are stable
    return A.Compose([A.Resize(IMG_H, IMG_W)])

# one item = (ultrasound image, head mask), both as float tensors
class HC18Dataset(Dataset):

    def __init__(self, raw_dir, processed_dir, split, train=False):
        # splits.csv (made by prepare_data.py) says which images belong to which set
        df = pd.read_csv(Path(processed_dir) / "splits.csv")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.raw = Path(raw_dir) / "training_set"
        self.masks = Path(processed_dir) / "masks"
        self.tf = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        fn = self.df.loc[i, "filename"]

        # read the scan and its mask as grayscale images
        img = cv2.imread(str(self.raw / fn), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.masks / fn), cv2.IMREAD_GRAYSCALE)

        # apply the same resize/augmentation to both so they stay aligned
        out = self.tf(image=img, mask=mask)

        # scale pixel values from 0..255 to -1..1, which is easier for the network
        img = out["image"].astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        # turn the mask back into strict 0/1 values since resizing can blur the edges
        mask = (out["mask"] > 127).astype(np.float32)

        # add a channel dimension: (H, W) -> (1, H, W), as pytorch expects
        return (torch.from_numpy(img)[None], torch.from_numpy(mask)[None])