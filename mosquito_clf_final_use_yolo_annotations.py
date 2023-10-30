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
        EarlyStopping(monitor="val_f1_score", mode="max", patience=5),
    ]


BATCH_SIZE = 16
HEAD_NUMBER = 7
FREEZE_BACKBONE = False
MODEL_NAME = "ViT-L-14"
PRETRAIN_DATASET = "datacomp_xl_s13b_b90k"
AUGMENTATION = "hca"
WARMUP_STEPS = 1000
EPOCHS = 15
LABEL_SMOOTHING = 0.1
CALLBACK = callbacks
IMG_SIZE = (224, 224)


img_dir = "../data_round_2/final"
ANNOTATIONS_CSV = "../data_round_2/phase2_train_v0_cleaned_yolo_best_annotations.csv"


for USE_EMA in [False, True]:
    ExperimentMosquitoClassifier(img_dir, ANNOTATIONS_CSV).run_cross_validation(
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
        CALLBACK,
        False,
        USE_EMA,
        False,
    )
