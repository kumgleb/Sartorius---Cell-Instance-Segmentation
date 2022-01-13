import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def rle2mask_by_index(idx: int, data_df: pd.DataFrame) -> np.ndarray:

    rle = data_df.loc[idx].annotation
    img_w = data_df.loc[idx].width
    img_h = data_df.loc[idx].height

    ## transforming the string into an array of shape (2, N)
    array = np.fromiter(rle.split(), dtype=np.uint)
    array = array.reshape((-1, 2)).T
    array[0] = array[0] - 1

    ## decompressing the rle encoding (ie, turning [3, 1, 10, 2] into [3, 4, 10, 11, 12])
    # for faster mask construction
    starts, lenghts = array
    mask_decompressed = np.concatenate(
        [np.arange(s, s + l, dtype=np.uint) for s, l in zip(starts, lenghts)]
    )

    ## Building the binary mask
    msk_img = np.zeros(img_w * img_h, dtype=np.uint8)
    msk_img[mask_decompressed] = 1
    msk_img = msk_img.reshape((img_h, img_w))

    return msk_img


def mask_by_id(img_id: str, data_df: pd.DataFrame) -> np.ndarray:

    data_ = data_df[data_df.id == img_id]
    img_w = data_.width.values[0]
    img_h = data_.height.values[0]

    mask = np.zeros((img_h, img_w))

    idxs = data_df[data_df.id == img_id].index

    for idx in idxs:
        mask_ = rle2mask_by_index(idx, data_df)
        mask += mask_

    return mask.clip(0, 1)


class CellDataset(Dataset):
    def __init__(
        self, imgs_folder: str, data_df: pd.DataFrame, img_ids: list, transforms=None
    ):
        super().__init__()
        self.data_df = data_df
        self.imgs_folder = imgs_folder
        self.transforms = transforms
        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        id_str = self.img_ids[idx]
        img_path = os.path.join(self.imgs_folder, f"{id_str}.png")
        img = np.asarray(Image.open(img_path), dtype="uint8")
        mask = mask_by_id(id_str, self.data_df)

        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        data = {"image": img, "mask": mask}

        return data, id_str
