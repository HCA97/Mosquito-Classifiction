from typing import List

import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
import pandas as pd

import src.classification as lc
import src.data_loader as dl

from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping

from src.experiments import ExperimentMosquitoClassifier


def callbacks() -> List[Callback]:
    return [
        ModelCheckpoint(
            monitor="val_f1_score",
            mode="max",
            save_top_k=2,
            save_last=False,
            save_weights_only=True,
            filename="{epoch}-{val_loss}-{val_f1_score}-{val_multiclass_accuracy}",
        ),
        EarlyStopping(monitor="val_f1_score", mode="max", patience=5),
    ]


img_dir = "../data_round_2/final"
val_annotations_csv = "../data_round_2/best_model_val_data_yolo_annotations.csv"
train_annotations_csv = "../data_round_2/best_model_train_data_yolo_annotations.csv"
inat_annotations_csv = "../inaturalist_crawler_data_all_cleaned_sub/inaturalist.csv"

train_df = pd.read_csv(train_annotations_csv)
inat_df = pd.read_csv(inat_annotations_csv)
inat_df = inat_df[inat_df["class_label"] != "culex"]
inat_df = inat_df[inat_df["class_label"] != "albopictus"]

train_df["img_fName"] = "../data_round_2/final/" + train_df["img_fName"]
inat_df["img_fName"] = (
    "../inaturalist_crawler_data_all_cleaned_sub/" + inat_df["img_fName"]
)

train_df = pd.concat([train_df, inat_df])
val_df = pd.read_csv(val_annotations_csv)

model = lc.MosquitoClassifier(
    bs=16,
    head_version=7,
    freeze_backbones=False,
    label_smoothing=0.1,
    data_aug="hca",
    epochs=15,
    max_steps=60000,
    use_ema=True,
)


train_dataloader, _ = ExperimentMosquitoClassifier(".", "").get_dataloaders(
    train_df,
    val_df,
    model.hparams.dataset,
    model.hparams.data_aug,
    model.hparams.bs,
    model.hparams.img_size,
    model.hparams.shift_box,
)

_, val_dataloader = ExperimentMosquitoClassifier(img_dir, "").get_dataloaders(
    train_df,
    val_df,
    model.hparams.dataset,
    model.hparams.data_aug,
    model.hparams.bs,
    model.hparams.img_size,
    model.hparams.shift_box,
)
th.set_float32_matmul_precision("high")
trainer = pl.Trainer(
    accelerator="gpu",
    precision="16-mixed",
    max_epochs=model.hparams.epochs,
    logger=True,
    deterministic=True,  # maybe we should add this
    callbacks=callbacks(),
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
