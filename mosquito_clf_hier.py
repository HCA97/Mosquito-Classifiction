import os

import pandas as pd
import torch as th

from src.experiments import ExperimentMosquitoClassifier


MODEL_NAME = "ViT-L-14"
PRETRAIN_DATASET = "datacomp_xl_s13b_b90k"
BATCH_SIZE = 16
HEAD_NUMBER = 7
FREEZE_BACKBONE = False
AUGMENTATION = "hca"
WARMUP_STEPS = 1000
EPOCHS = 8
LABEL_SMOOTHING = 0.1


img_dir = "../data_round_2/final"
annotations_csv = "../data_round_2/phase2_train_v0_cleaned.csv"

df = pd.read_csv(annotations_csv)
# GENUS
df_genus = df[
    (df.class_label == "anopheles")
    | (df.class_label == "culex")
    | (df.class_label == "culiseta")
]


genus_csv = os.path.join(os.path.split(annotations_csv)[0], "genus_only_cleaned.csv")
df_genus.to_csv(genus_csv, index=False)

exp = ExperimentMosquitoClassifier(
    img_dir,
    genus_csv,
    class_dict={
        "anopheles": th.tensor([1, 0, 0], dtype=th.float),
        "culex": th.tensor([0, 1, 0], dtype=th.float),
        "culiseta": th.tensor([0, 0, 1], dtype=th.float),
    },
)

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
)

# SPECIES
df_species = df[
    (df.class_label == "aegypti")
    | (df.class_label == "albopictus")
    | (df.class_label == "japonicus/koreicus")
]

species_csv = os.path.join(
    os.path.split(annotations_csv)[0], "species_only_cleaned.csv"
)
df_species.to_csv(species_csv, index=False)

exp = ExperimentMosquitoClassifier(
    img_dir,
    species_csv,
    class_dict={
        "aegypti": th.tensor([1, 0, 0], dtype=th.float),
        "albopictus": th.tensor([0, 1, 0], dtype=th.float),
        "japonicus/koreicus": th.tensor([0, 0, 1], dtype=th.float),
    },
)

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
)
