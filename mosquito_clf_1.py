from typing import List
import random

import torch as th
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import pandas as pd

import src.classification as lc
import src.data_loader as dl

img_dir = "../data/train"
annotations_csv = "../data/train.csv"
class_dict = {
    "albopictus": th.tensor([1, 0, 0, 0, 0, 0], dtype=th.float),
    "culex": th.tensor([0, 1, 0, 0, 0, 0], dtype=th.float),
    "japonicus/koreicus": th.tensor([0, 0, 1, 0, 0, 0], dtype=th.float),
    "culiseta": th.tensor([0, 0, 0, 1, 0, 0], dtype=th.float),
    "anopheles": th.tensor([0, 0, 0, 0, 1, 0], dtype=th.float),
    "aegypti": th.tensor([0, 0, 0, 0, 0, 1], dtype=th.float),
}


def get_dataloaders(model_name: str, data_aug: str, bs: int) -> List[DataLoader]:
    global img_dir, annotations_csv, class_dict

    transform = dl.pre_process(model_name)

    annotations_df = pd.read_csv(annotations_csv)
    train_df = annotations_df.sample(frac=0.8, random_state=200)
    val_df = annotations_df.drop(train_df.index)

    train_dataset = dl.SimpleClassificationDataset(
        train_df,
        img_dir,
        class_dict,
        transform,
        dl.aug(data_aug),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=6,
        drop_last=True,
    )

    val_dataset = dl.SimpleClassificationDataset(
        val_df,
        img_dir,
        class_dict,
        transform,
        dl.aug("resize"),
        class_balance=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=6,
    )

    return train_dataloader, val_dataloader


def get_callbacks() -> List[Callback]:
    return [
        ModelCheckpoint(
            monitor="val_multiclass_accuracy",
            mode="max",
            save_top_k=2,
            save_last=True,
            filename="{epoch}-{val_loss}-{val_accuracy}-{val_multiclass_accuracy}",
        ),
    ]


def train(
    model_name: str,
    dataset: str,
    bs: int,
    head_version: int,
    data_aug: str,
    freeze_backbones: bool = False,
    warm_up_steps: int = 2000,
):
    train_dataloader, val_dataloader = get_dataloaders(model_name, data_aug, bs)

    th.set_float32_matmul_precision("high")
    model = lc.MosquitoClassifier(
        model_name=model_name,
        dataset=dataset,
        freeze_backbones=freeze_backbones,
        head_version=head_version,
        warm_up_steps=warm_up_steps,
        bs=bs,
        data_aug=data_aug,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        precision="16-mixed",
        max_epochs=5,
        logger=True,
        callbacks=get_callbacks(),
    )

    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


# params = []
# for data_aug in ["image_net", "happy_whale", "hca"]:
#     for fb in [False]:
#         for warm_up_steps in [1000, 1500]:
#             for head_version in [2]:
#                 for model in [
#                     ["ViT-B-16", "datacomp_l_s1b_b8k", 64],
#                     ["ViT-L-14", "datacomp_xl_s13b_b90k", 64],
#                 ]:
#                     param = model + [head_version, data_aug, fb, warm_up_steps]
#                     params.append(param)

# for data_aug in ["image_net", "happy_whale", "hca"]:
#     for fb in [True]:
#         for warm_up_steps in [100, 0]:
#             for head_version in [2]:
#                 for model in [
#                     ["ViT-B-16", "datacomp_l_s1b_b8k", 64],
#                     ["ViT-L-14", "datacomp_xl_s13b_b90k", 64],
#                 ]:
#                     param = model + [head_version, data_aug, fb, warm_up_steps]
#                     params.append(param)

# random.shuffle(params)

# print(f"Total experiments {len(params)}")
# for param in params[:50]:
#     print("Params:", param)
#     train(*param)


train("ViT-L-14", "datacomp_xl_s13b_b90k", 64, 4, "hca", False, 1000)
