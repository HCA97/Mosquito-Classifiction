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
            save_weights_only=True,
            filename="{epoch}-{val_loss}-{val_f1_score}-{val_multiclass_accuracy}",
        ),
        EarlyStopping(monitor="val_f1_score", mode="max", patience=2),
    ]


BATCH_SIZE = 16
HEAD_NUMBER = 7
FREEZE_BACKBONE = False
AUGMENTATION = "hca"
WARMUP_STEPS = 1000
EPOCHS = 8
LABEL_SMOOTHING = 0.1

img_dir = "../data_round_2/final"
annotations_csv = "../data_round_2/phase2_train_v0_cleaned.csv"

exp = ExperimentMosquitoClassifier(img_dir, annotations_csv)

for MODEL_NAME, PRETRAIN_DATASET in [
    # ("covnext_large", "imagenet"),
    # ("covnext_xlarge", "imagenet"),
    # ("convnext_large_d", "laion2b_s26b_b102k_augreg"),
    ("convnext_large_d.trunk", "laion2b_s26b_b102k_augreg"),
    ("convnext_large_d_320", "laion2b_s29b_b131k_ft_soup"),
    ("convnext_large_d_320.trunk", "laion2b_s29b_b131k_ft_soup"),
]:
    for IMG_SIZE in [
        (256, 256),
        (320, 320),
    ]:
        # if (
        #     PRETRAIN_DATASET == "laion2b_s26b_b102k_augreg"
        #     and MODEL_NAME == "convnext_large_d"
        #     and IMG_SIZE == (256, 256)
        # ):
        #     continue

        exp.run(
            MODEL_NAME,
            PRETRAIN_DATASET,
            BATCH_SIZE,
            HEAD_NUMBER,
            AUGMENTATION,
            FREEZE_BACKBONE,
            WARMUP_STEPS,
            EPOCHS,
            LABEL_SMOOTHING,
            IMG_SIZE,
            callbacks,
        )
