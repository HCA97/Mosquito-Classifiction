from typing import List

import torch
import pytorch_lightning as pl

from src.classification import MosquitoClassifier


def model_soup(path_list: List[str], save_path: str):
    # Load models weights
    weight_list = []

    for path in path_list:
        print(f"Loading {path}...")
        model = MosquitoClassifier.load_from_checkpoint(
            path, map_location=torch.device("cpu")
        )
        weight_list.append(model.state_dict())

    # Average weights
    state_dict = {}

    for k in weight_list[0]:
        try:
            w = torch.stack([v[k] for v in weight_list]).mean(0)
            print(f"Mean: Layer {k}...")
        except:
            w = torch.stack([v[k] for v in weight_list]).sum(0)
            print(f"Sum: Layer {k}...")

        state_dict[k] = w
    # state_dict = dict(
    #     (k, torch.stack([v[k] for v in weight_list]).mean(0)) for k in weight_list[0]
    # )
    model.load_state_dict(state_dict)

    trainer = pl.Trainer()
    trainer.strategy.connect(model)
    trainer.save_checkpoint(save_path)


paths = [
    "./Mosquito-Classifiction/lightning_logs/version_35/checkpoints/epoch=4-val_loss=0.2117716670036316-val_f1_score=0.6801754832267761-val_multiclass_accuracy=0.6670776009559631.ckpt",
    "./Mosquito-Classifiction/lightning_logs/version_34/checkpoints/epoch=3-val_loss=0.23519441485404968-val_f1_score=0.7126025557518005-val_multiclass_accuracy=0.7606703639030457.ckpt",
    "./Mosquito-Classifiction/lightning_logs/version_33/checkpoints/epoch=4-val_loss=0.1846015751361847-val_f1_score=0.7405803799629211-val_multiclass_accuracy=0.7194995284080505.ckpt",
    "./Mosquito-Classifiction/lightning_logs/version_32/checkpoints/epoch=4-val_loss=0.26963797211647034-val_f1_score=0.6824541091918945-val_multiclass_accuracy=0.7079151272773743.ckpt",
    "./Mosquito-Classifiction/lightning_logs/version_31/checkpoints/epoch=4-val_loss=0.23801635205745697-val_f1_score=0.7440996170043945-val_multiclass_accuracy=0.7687538266181946.ckpt",
]
save_path = "ViT-B-16-k=5_soup.ckpt"

model_soup(paths, save_path)
