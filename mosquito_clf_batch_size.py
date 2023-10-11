import random
from typing import List

from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from src.experiments import ExperimentMosquitoClassifier


def callbacks() -> List[Callback]:
    return [
        ModelCheckpoint(
            monitor="val_f1_score",
            mode="max",
            save_top_k=1,
            save_last=False,
            filename="{epoch}-{val_loss}-{val_f1_score}-{val_multiclass_accuracy}",
        ),
        EarlyStopping(monitor="val_f1_score", mode="max", patience=2),
    ]


img_dir = "../data_round_2/final"
annotations_csv = "../data_round_2/phase2_train_v0.csv"

exp = ExperimentMosquitoClassifier(img_dir, annotations_csv)

params = []
for epoch in [20]:
    for head_version in [4, 6, 5]:
        for bs in [8, 16, 32, 64]:
            param = [
                "ViT-L-14",
                "datacomp_xl_s13b_b90k",
                bs,
                head_version,
                "hca",
                False,
                1000,
                epoch,
                0.0,
                (224, 224),
                callbacks,
            ]
            params.append(param)

random.shuffle(params)

print(f"Total experiments {len(params)}")
for param in params[:50]:
    print("Params:", param)
    exp.run(*param)
