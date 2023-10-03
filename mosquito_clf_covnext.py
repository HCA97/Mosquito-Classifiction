import random

from src.experiments import ExperimentMosquitoClassifier


MODEL_NAME = "convnext_large_d"
PRETRAIN_DATASET = "laion2b_s26b_b102k_augreg"
BATCH_SIZE = 16
HEAD_NUMBER = 7
FREEZE_BACKBONE = False
AUGMENTATION = "hca"
WARMUP_STEPS = 1000
EPOCHS = 8
LABEL_SMOOTHING = 0.1
IMG_SIZE = (256, 256)

img_dir = "../data_round_2/final"
annotations_csv = "../data_round_2/phase2_train_v0_cleaned.csv"

exp = ExperimentMosquitoClassifier(img_dir, annotations_csv)

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
)
