import os
import logging
from typing import Optional, Union

import torch as th
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms as T
import albumentations as A


def pre_process(_: str) -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def aug(data_aug: str = "image_net") -> T.Compose:
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(
                size=(224, 224),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
    )
    if data_aug == "image_net":
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                T.Resize(
                    size=(224, 224),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )

    elif data_aug == "hca":
        aug8p3 = A.OneOf(
            [
                A.Sharpen(p=0.3),
                A.ToGray(p=0.3),
                A.CLAHE(p=0.3),
            ],
            p=0.5,
        )

        blur = A.OneOf(
            [
                A.GaussianBlur(p=0.3),
                A.MotionBlur(p=0.3),
            ],
            p=0.5,
        )

        transform = A.Compose(
            [
                A.ShiftScaleRotate(
                    rotate_limit=45,
                    scale_limit=0.1,
                    border_mode=cv2.BORDER_REFLECT,
                    interpolation=cv2.INTER_CUBIC,
                    p=0.5,
                ),
                A.Resize(224, 224, cv2.INTER_CUBIC),
                aug8p3,
                blur,
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    elif data_aug == "aug_mix":
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.AugMix(),
                T.Resize(
                    size=(224, 224),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )
    elif data_aug == "happy_whale":
        aug8p3 = A.OneOf(
            [
                A.Sharpen(p=0.3),
                A.ToGray(p=0.3),
                A.CLAHE(p=0.3),
            ],
            p=0.5,
        )

        transform = A.Compose(
            [
                A.ShiftScaleRotate(
                    rotate_limit=15,
                    scale_limit=0.1,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
                A.Resize(224, 224, cv2.INTER_CUBIC),
                aug8p3,
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )

    elif data_aug == "cut_out":
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ImageCompression(quality_lower=99, quality_upper=100),
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.7,
                ),
                A.Resize(224, 224, cv2.INTER_CUBIC),
                A.Cutout(
                    max_h_size=int(224 * 0.4),
                    max_w_size=int(224 * 0.4),
                    num_holes=1,
                    p=0.5,
                ),
            ]
        )
    elif data_aug == "clip":
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Resize(
                    size=(224, 224),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )
    elif data_aug == "clip+image_net":
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
                T.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Resize(
                    size=(224, 224),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )

    return transform


def read_image_cv2(f_name: str, gray_scale: bool = False) -> np.ndarray:
    img = cv2.imread(
        f_name, cv2.IMREAD_ANYCOLOR if not gray_scale else cv2.IMREAD_GRAYSCALE
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def class_balancing(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.class_label.value_counts().to_dict()
    max_label = max(list(counts.items()), key=lambda x: x[1])

    for key, value in counts.items():
        if key == max_label[0]:
            continue

        df_label = df[df.class_label == key].sample(
            n=max_label[1] - value, replace=True
        )
        df = pd.concat([df, df_label])

    return df


class SimpleClassificationDataset(Dataset):
    def __init__(
        self,
        annotations_df: pd.DataFrame,
        img_dir: str,
        class_dict: dict,
        transform: Optional[T.Compose] = None,
        data_augment: Optional[Union[T.Compose, A.Compose]] = None,
        class_balance: bool = True,
    ):
        self.df = annotations_df
        if class_balance:
            self.df = class_balancing(annotations_df)

        self.img_dir = img_dir
        self.class_dict = class_dict
        self.transform = transform
        self.data_augment = data_augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cv2.setNumThreads(6)

        f_name, _, _, x_tl, y_tl, x_br, y_br, label = self.df.iloc[idx]

        img = read_image_cv2(os.path.join(self.img_dir, f_name))
        img_ = img[y_tl:y_br, x_tl:x_br, :]
        if img_.shape[0] * img_.shape[1] != 0:
            img = img_

        if self.data_augment:
            if isinstance(self.data_augment, A.Compose):
                img = self.data_augment(image=img)["image"]
            else:
                img = self.data_augment(img)

        if self.transform:
            img = self.transform(img)

        if self.class_dict:
            label = self.class_dict[label]
        return {"img": img, "label": label}


if __name__ == "__main__":
    import torch as th
    import torchvision.transforms as T
    import albumentations as A
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))

    img_dir = "../data/train"
    annotations_csv = "../data/train.csv"
    annotations_df = pd.read_csv(annotations_csv)
    class_dict = {
        "albopictus": th.tensor([1, 0, 0, 0, 0, 0], dtype=th.long),
        "culex": th.tensor([0, 1, 0, 0, 0, 0], dtype=th.long),
        "japonicus/koreicus": th.tensor([0, 0, 1, 0, 0, 0], dtype=th.long),
        "culiseta": th.tensor([0, 0, 0, 1, 0, 0], dtype=th.long),
        "anopheles": th.tensor([0, 0, 0, 0, 1, 0], dtype=th.long),
        "aegypti": th.tensor([0, 0, 0, 0, 0, 1], dtype=th.long),
    }

    transform = pre_process("")

    data_augmentation = aug("image_net")

    ds = SimpleClassificationDataset(
        annotations_df=annotations_df,
        img_dir=img_dir,
        class_dict=class_dict,
        transform=transform,
        data_augment=data_augmentation,
    )
    for i in range(10):
        res = ds[i]
        img = res["img"]

        img_bbox = th.tensor(
            255 * (img - img.min()) / (img.max() - img.min()), dtype=th.uint8
        )
        show(img_bbox)
        plt.show()
