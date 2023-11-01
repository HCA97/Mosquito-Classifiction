from typing import List, Dict, Callable, Tuple

import torch as th
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd

import src.classification as lc
import src.data_loader as dl


def _default_callbacks() -> List[Callback]:
    return [
        ModelCheckpoint(
            monitor="val_f1_score",
            mode="max",
            save_top_k=2,
            save_last=False,
            save_weights_only=True,
            filename="{epoch}-{val_loss}-{val_f1_score}-{val_multiclass_accuracy}",
        ),
    ]


CLASS_DICT = {
    "albopictus": th.tensor([1, 0, 0, 0, 0, 0], dtype=th.float),
    "culex": th.tensor([0, 1, 0, 0, 0, 0], dtype=th.float),
    "japonicus/koreicus": th.tensor([0, 0, 1, 0, 0, 0], dtype=th.float),
    "culiseta": th.tensor([0, 0, 0, 1, 0, 0], dtype=th.float),
    "anopheles": th.tensor([0, 0, 0, 0, 1, 0], dtype=th.float),
    "aegypti": th.tensor([0, 0, 0, 0, 0, 1], dtype=th.float),
}


class ExperimentMosquitoClassifier:
    def __init__(
        self,
        img_dir: str,
        annotations_csv: str,
        class_dict: Dict[str, th.Tensor] = CLASS_DICT,
    ):
        self.img_dir = img_dir
        self.annotations_csv = annotations_csv
        self.class_dict = class_dict

    def get_dataloaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        dataset_name: str,
        data_aug: str,
        bs: int,
        img_size: Tuple[int, int] = (224, 224),
        shift_box: bool = False,
    ) -> List[DataLoader]:
        transform = dl.pre_process(dataset_name)

        train_dataset = dl.SimpleClassificationDataset(
            train_df,
            self.img_dir,
            self.class_dict,
            transform,
            dl.aug(data_aug, img_size),
            shift_box=shift_box,
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
            self.img_dir,
            self.class_dict,
            transform,
            dl.aug("resize", img_size),
            class_balance=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=6,
        )

        return train_dataloader, val_dataloader

    def run(
        self,
        model_name: str,
        dataset: str,
        bs: int,
        head_version: int,
        data_aug: str,
        freeze_backbones: bool = False,
        warm_up_steps: int = 2000,
        epochs: int = 5,
        label_smoothing: float = 0.0,
        img_size: Tuple[int, int] = (224, 224),
        create_callbacks: Callable[[], List[Callback]] = _default_callbacks,
        use_same_split_as_yolo: bool = False,
        use_ema: bool = False,
        shift_box: bool = False,
    ):
        annotations_df = pd.read_csv(self.annotations_csv)
        train_df, val_df = train_test_split(
            annotations_df,
            test_size=0.2,
            stratify=annotations_df["class_label"],
            random_state=200,
        )
        if use_same_split_as_yolo:
            df_img_label = annotations_df[
                ["img_fName", "class_label"]
            ].drop_duplicates()

            _train_data, _val_data = train_test_split(
                df_img_label,
                test_size=0.2,
                stratify=df_img_label["class_label"],
                random_state=200,
            )

            _train_list = list(set(_train_data["img_fName"]))
            _val_list = list(set(_val_data["img_fName"]))

            train_df = annotations_df[annotations_df["img_fName"].isin(_train_list)]
            val_df = annotations_df[annotations_df["img_fName"].isin(_val_list)]

        train_dataloader, val_dataloader = self.get_dataloaders(
            train_df, val_df, dataset, data_aug, bs, img_size, shift_box
        )

        th.set_float32_matmul_precision("high")
        model = lc.MosquitoClassifier(
            n_classes=len(self.class_dict),
            model_name=model_name,
            dataset=dataset,
            freeze_backbones=freeze_backbones,
            head_version=head_version,
            warm_up_steps=warm_up_steps,
            bs=bs,
            data_aug=data_aug,
            epochs=epochs,
            label_smoothing=label_smoothing,
            img_size=img_size,
            use_ema=use_ema,
            use_same_split_as_yolo=use_same_split_as_yolo,
            shift_box=shift_box,
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            precision="16-mixed",
            max_epochs=epochs,
            logger=True,
            deterministic=True,  # maybe we should add this
            callbacks=create_callbacks(),
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def run_cross_validation(
        self,
        model_name: str,
        dataset: str,
        bs: int,
        head_version: int,
        data_aug: str,
        freeze_backbones: bool = False,
        warm_up_steps: int = 2000,
        epochs: int = 5,
        label_smoothing: float = 0.0,
        img_size: Tuple[int, int] = (224, 224),
        create_callbacks: Callable[[], List[Callback]] = _default_callbacks,
        use_same_split_as_yolo: bool = False,
        use_ema: bool = False,
        shift_box: bool = False,
    ):
        annotations_df = pd.read_csv(self.annotations_csv)
        skf = StratifiedKFold(n_splits=5, random_state=200)

        for _, (train_index, val_index) in enumerate(
            skf.split(annotations_df, annotations_df.class_label)
        ):
            train_df = annotations_df.iloc[train_index]
            val_df = annotations_df.iloc[val_index]

            train_dataloader, val_dataloader = self.get_dataloaders(
                train_df, val_df, dataset, data_aug, bs, img_size, shift_box
            )

            th.set_float32_matmul_precision("high")
            model = lc.MosquitoClassifier(
                n_classes=len(self.class_dict),
                model_name=model_name,
                dataset=dataset,
                freeze_backbones=freeze_backbones,
                head_version=head_version,
                warm_up_steps=warm_up_steps,
                bs=bs,
                data_aug=data_aug,
                epochs=epochs,
                label_smoothing=label_smoothing,
                img_size=img_size,
                use_ema=use_ema,
                use_same_split_as_yolo=use_same_split_as_yolo,
                shift_box=shift_box,
            )
            trainer = pl.Trainer(
                accelerator="gpu",
                precision="16-mixed",
                max_epochs=epochs,
                logger=True,
                callbacks=create_callbacks(),  # if I pass it as list of callbacks it doesn't work
                deterministic=True,  # maybe we should add this
                # TODO: we need some naming convention
            )

            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
