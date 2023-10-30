import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import yaml
import sys

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


img_dir = "../../data_round_2/final"
annotation_csv = "../../data_round_2/phase2_train_v0.csv"

class_dict = {"genus": 1, "species": 0}

output_dir = "../../data_yolo_two_class"
yaml_file = "yolo_config_mos_two_class.yml"


def class_balancing(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.class_label.value_counts().to_dict()
    max_label = max(list(counts.items()), key=lambda x: x[1])

    for key, value in counts.items():
        if key == max_label[0]:
            continue

        df_label = df[df.class_label == key].sample(
            n=max_label[1] - value, replace=True
        )
        df = pd.concat([df, df_label])

    return df


def convert_2_yolo_boxxes(img_shape: tuple, bbox: tuple) -> tuple:
    img_w, img_h = img_shape
    x_tl, y_tl, x_br, y_br = bbox

    box_w = x_br - x_tl
    box_h = y_br - y_tl

    x_c = (x_tl + x_br) / 2
    y_c = (y_tl + y_br) / 2

    return (x_c / img_w, y_c / img_h, box_w / img_w, box_h / img_h)


def create_folders(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)


def create_yolo_folder(df: pd.DataFrame, folder_name: str, start_index: int = 0):
    def _loop(i: int, rows: pd.DataFrame):
        nonlocal folder_name, df, start_index
        global img_dir, output_dir

        label_path = os.path.join(
            output_dir, "labels", folder_name, f"{i + start_index}.txt"
        )

        with open(label_path, "w") as f:
            for j in range(len(rows)):
                f_name, w, h, x_tl, y_tl, x_br, y_br, label = rows.iloc[j]
                bbox = convert_2_yolo_boxxes((w, h), (x_tl, y_tl, x_br, y_br))

                _label = 1 if label.lower() in ["anopheles", "culex", "culiseta"] else 0

                if j == (len(rows) - 1):
                    f.write(f"{_label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")
                else:
                    f.write(f"{_label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

        src_path = os.path.join(img_dir, f_name)
        dst_path = os.path.join(
            output_dir, "images", folder_name, f"{i + start_index}.jpeg"
        )
        shutil.copy(src_path, dst_path)

    df_groupped = df.groupby("img_fName", group_keys=True).apply(lambda x: x)
    with ThreadPoolExecutor(10) as exe:
        jobs = []
        for i, img_fName in enumerate(set(df_groupped["img_fName"])):
            rows = df_groupped[df_groupped.img_fName == img_fName]
            jobs.append(exe.submit(_loop, i, rows))

        for job in tqdm(jobs):
            job.result()


if __name__ == "__main__":
    annotations_df = pd.read_csv(annotation_csv)
    train_df, val_df = train_test_split(
        annotations_df,
        test_size=0.2,
        stratify=annotations_df["class_label"],
        random_state=200,
    )

    create_folders(output_dir)
    create_yolo_folder(train_df, "train")
    create_yolo_folder(val_df, "val", len(train_df))

    config = {
        "path": output_dir,
        "train": "./images/train",
        "val": "./images/val",
        "names": dict((v, k) for k, v in class_dict.items()),
    }

    with open(yaml_file, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
