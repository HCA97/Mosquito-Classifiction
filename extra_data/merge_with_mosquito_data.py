import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import pandas as pd

output_dir = "../../data_round_2+kaggle+human-skin"
datasets = [
    (
        "../../Mosquito-on-human-skin/Aedes_aegypti_landing_stabilityai_x4",
        "../../Mosquito-on-human-skin/Aedes_aegypti_landing_stabilityai_x4.csv",
    ),
    (
        "../../Mosquito-on-human-skin/Aedes_aegypti_smashed_stabilityai_x4",
        "../../Mosquito-on-human-skin/Aedes_aegypti_smashed_stabilityai_x4.csv",
    ),
    (
        "../../mosqutio_kaggle/aegypti_0",
        "../../mosqutio_kaggle/aegypti_0.csv",
    ),
    (
        "../../data_round_2/final",
        "../../data_round_2/phase2_train_v0.csv",
    ),
]


def move_data(img_dir: str, output_dir: str, modify_name: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    folder_name = os.path.split(img_dir)[-1]

    with ThreadPoolExecutor(10) as exe:
        jobs = []
        for img_path in os.listdir(img_dir):
            src_path = os.path.join(img_dir, img_path)
            dst_path = os.path.join(output_dir, img_path)
            if modify_name:
                dst_path = os.path.join(output_dir, f"{folder_name}_{img_path}")
            jobs.append(exe.submit(shutil.copy, src_path, dst_path))

        for job in tqdm(jobs):
            job.result()


dfs = []

for img_dir, csv_path in datasets:
    df = pd.read_csv(csv_path)

    dfs.append(df)

    move_data(
        img_dir,
        os.path.join(output_dir, "images"),
        img_dir != "../../data_round_2/final",
    )


df = pd.concat(dfs)
df.to_csv(os.path.join(output_dir, "combined.csv"), index=False)
