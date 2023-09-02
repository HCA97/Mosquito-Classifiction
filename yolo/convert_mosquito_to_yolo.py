import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import yaml
import sys

import pandas as pd
from tqdm import tqdm


img_dir = "../../data_round_2/final"
annotation_csv = "../../data_round_2/phase2_train_v0.csv"
class_dict = {
    "albopictus": 0,
    "culex": 1,
    "japonicus/koreicus": 2,
    "culiseta": 3,
    "anopheles": 4,
    "aegypti": 5,
}


output_dir = "../../data_yolo"
yaml_file = "yolo_config_mos.yml"


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
    def _loop(i: int):
        nonlocal folder_name, df, start_index
        global img_dir, output_dir, class_dict

        f_name, w, h, x_tl, y_tl, x_br, y_br, label = df.iloc[i]
        src_path = os.path.join(img_dir, f_name)
        dst_path = os.path.join(
            output_dir, "images", folder_name, f"{i + start_index}.jpeg"
        )
        shutil.copy(src_path, dst_path)

        bbox = convert_2_yolo_boxxes((w, h), (x_tl, y_tl, x_br, y_br))

        label_path = os.path.join(
            output_dir, "labels", folder_name, f"{i + start_index}.txt"
        )
        with open(label_path, "w") as f:
            f.write(f"{class_dict[label]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

    with ThreadPoolExecutor(10) as exe:
        jobs = []
        for i in range(len(df)):
            jobs.append(exe.submit(_loop, i))

        for job in tqdm(jobs):
            job.result()


if __name__ == "__main__":
    df = pd.read_csv(annotation_csv)

    train_df = df.sample(frac=0.8, random_state=200)
    val_df = df.drop(train_df.index)

    # not yet
    if sys.argv[1] == "balance":
        print("balance dataset")
        train_df = class_balancing(train_df)
        output_dir += "_balance"
        yaml_file = "yolo_config_mos_balance.yml"

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
