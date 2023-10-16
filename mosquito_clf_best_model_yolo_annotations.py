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
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
            filename="{epoch}-{val_loss}-{val_f1_score}-{val_multiclass_accuracy}",
        ),
        EarlyStopping(monitor="val_f1_score", mode="max", patience=2),
    ]


img_dir = "../data_round_2/final"
val_annotations_csv = "../data_round_2/best_model_val_data_yolo_annotations.csv"
train_annotations_csv = "../data_round_2/best_model_train_data_yolo_annotations.csv"

checkpoint_path = "../mosquitoalert-2023-phase2-starter-kit/my_models/clip_weights/epoch=7-val_loss=0.7274188995361328-val_f1_score=0.8357369303703308-val_multiclass_accuracy=0.8175098896026611.ckpt"

train_df = pd.read_csv(train_annotations_csv)
val_df = pd.read_csv(val_annotations_csv)

model = lc.MosquitoClassifier.load_from_checkpoint(
    checkpoint_path, head_version=7, freeze_backbones=True, label_smoothing=0.1
)


train_dataloader, val_dataloader = ExperimentMosquitoClassifier(
    img_dir, ""
).get_dataloaders(
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
    max_epochs=5,
    logger=True,
    deterministic=True,  # maybe we should add this
    callbacks=callbacks(),
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
