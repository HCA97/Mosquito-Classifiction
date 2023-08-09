from torch.utils.data import DataLoader
import pandas as pd
import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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
callbacks = [
    ModelCheckpoint(
        every_n_epochs=1,
        save_top_k=-1,
        filename="{epoch}-val_loss={val_loss}-val_acc={val_accuracy}",
    ),
    EarlyStopping(monitor="val_loss"),
]


def train(model_name: str, dataset: str, data_aug: str, bs: int):
    global img_dir, annotations_csv, class_dict, callbacks

    annotations_df = pd.read_csv(annotations_csv)

    transform = dl.pre_process(model_name)

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
        persistent_workers=True,
        pin_memory=True,
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
        persistent_workers=True,
        pin_memory=True,
    )

    th.set_float32_matmul_precision("high")
    model = lc.MosquitoClassifier(model_name=model_name, dataset=dataset)
    trainer = pl.Trainer(
        accelerator="gpu",
        precision="16-mixed",
        max_epochs=10,
        logger=True,
        callbacks=callbacks,
    )

    trainer.fit(
        model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


params = [
    ("ViT-B-16", "datacomp_l_s1b_b8k", "image_net", 64),
    ("ViT-B-16", "datacomp_l_s1b_b8k", "happy_whale", 64),
    ("ViT-L-14", "datacomp_xl_s13b_b90k", "image_net", 64),
    ("ViT-L-14", "datacomp_xl_s13b_b90k", "happy_whale", 64),
    ("ViT-H-14", "laion2b_s32b_b79k", "image_net", 16),
    ("ViT-H-14", "laion2b_s32b_b79k", "happy_whale", 16),
    ("ViT-B-16", "datacomp_l_s1b_b8k", "cut_out", 64),
    ("ViT-L-14", "datacomp_xl_s13b_b90k", "cut_out", 64),
    ("ViT-H-14", "laion2b_s32b_b79k", "cut_out", 16),
]

for param in params:
    train(*param)
