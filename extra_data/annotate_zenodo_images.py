import csv
import os

from PIL import Image
import numpy as np


def annotate_zenodo_images(img_dir: str, save_path: str):
    folder_name = os.path.split(img_dir)[-1]
    data = []

    for img_path in os.listdir(img_dir):
        if os.path.splitext(img_path)[-1] in [".png", ".jpeg", ".jpg"]:
            print(f"Processing {img_path}...")
            image = Image.open(os.path.join(img_dir, img_path)).convert("RGB")

            if 'aegypti' in img_path:
                class_label = "aegypti"
            elif 'anopheles' in img_path:
                class_label = "anopheles"
            elif 'culex' in img_path:
                class_label = "culex"
            else:
                raise ValueError(f"Class {img_path} not know.")

            width, height = image.size
            boxes = [0, 0, width, height]
            data.append(
                {
                    "img_fName": f"{folder_name}_{img_path}",
                    "img_w": width,
                    "img_h": height,
                    "bbx_xtl": int(boxes[0]),
                    "bbx_ytl": int(boxes[1]),
                    "bbx_xbr": int(boxes[2]),
                    "bbx_ybr": int(boxes[3]),
                    "class_label": class_label,
                }
            )

    with open(save_path, "w") as csvfile:
        fieldnames = [
            "img_fName",
            "img_w",
            "img_h",
            "bbx_xtl",
            "bbx_ytl",
            "bbx_xbr",
            "bbx_ybr",
            "class_label",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


annotate_zenodo_images(
    "zenodo/images",
    "zenodo/zenodo_annotations.csv"
)
