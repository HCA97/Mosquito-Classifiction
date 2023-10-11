import random

from src.experiments import ExperimentMosquitoClassifier

img_dir = "../data_round_2/final"
annotations_csv = "../data_round_2/phase2_train_v0.csv"

exp = ExperimentMosquitoClassifier(img_dir, annotations_csv)

params = []
for data_aug in ["image_net", "happy_whale", "hca"]:
    for fb in [False]:
        for warm_up_steps in [1000, 1500]:
            for head_version in [2]:
                for model in [
                    ["ViT-B-16", "datacomp_l_s1b_b8k", 64],
                    ["ViT-L-14", "datacomp_xl_s13b_b90k", 64],
                ]:
                    param = model + [head_version, data_aug, fb, warm_up_steps]
                    params.append(param)

for data_aug in ["image_net", "happy_whale", "hca"]:
    for fb in [True]:
        for warm_up_steps in [100, 0]:
            for head_version in [2]:
                for model in [
                    ["ViT-B-16", "datacomp_l_s1b_b8k", 64],
                    ["ViT-L-14", "datacomp_xl_s13b_b90k", 64],
                ]:
                    param = model + [head_version, data_aug, fb, warm_up_steps]
                    params.append(param)

random.shuffle(params)

print(f"Total experiments {len(params)}")
for param in params[:50]:
    print("Params:", param)
    exp.run(*param)


# Test
# exp.run("ViT-B-16", "datacomp_l_s1b_b8k", 64, 2, "hca", False, 1000, epochs=5)
# exp.run("ViT-B-16", "datacomp_l_s1b_b8k", 64, 2, "hca", False, 1000, epochs=1)
# from pytorch_lightning.callbacks import ModelCheckpoint

# exp.run_cross_validation(
#     "ViT-B-16",
#     "datacomp_l_s1b_b8k",
#     64,
#     2,
#     "hca",
#     False,
#     1000,
#     epochs=5,
#     n_splits=5,
#     create_callbacks=lambda: [
#         ModelCheckpoint(
#             monitor="val_f1_score",
#             mode="max",
#             save_top_k=1,
#             save_last=False,
#             filename="{epoch}-{val_loss}-{val_f1_score}-{val_multiclass_accuracy}",
#         ),
#     ],
# )
