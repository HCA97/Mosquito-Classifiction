import csv
import os

import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np


def annotate_images(img_dir: str, save_path: str, class_label: str):
    owl_processor = OwlViTProcessor.from_pretrained(
        "google/owlvit-base-patch32", cache_dir="models/owl/"
    )
    owl_model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32", cache_dir="models/owl/"
    ).cuda()
    folder_name = os.path.split(img_dir)[-1]
    data = []

    for img_path in os.listdir(img_dir):
        if os.path.splitext(img_path)[-1] in [".png", ".jpeg", ".jpg"]:
            print(f"Processing {img_path}...")
            image = Image.open(os.path.join(img_dir, img_path)).convert("RGB")
            with torch.no_grad():
                texts = [["a photo of a mosquito"]]
                inputs = owl_processor(
                    text=texts, images=image, return_tensors="pt"
                ).to("cuda")
                outputs = owl_model(**inputs)
                # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
                target_sizes = torch.Tensor([image.size[::-1]]).to("cuda")

                # Convert outputs (bounding boxes and class logits) to COCO API
                results = owl_processor.post_process_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=0.01
                )

                i = 0  # Retrieve predictions for the first image for the corresponding text queries
                text = texts[i]
                boxes, scores, labels = (
                    results[i]["boxes"].cpu().numpy(),
                    results[i]["scores"].cpu().numpy(),
                    results[i]["labels"].cpu().numpy(),
                )
                best_score_index = np.argmax(scores)
                boxes, scores, labels = (
                    boxes[best_score_index],
                    scores[best_score_index],
                    labels[best_score_index],
                )

            width, height = image.size
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


annotate_images(
    "../Mosquito-on-human-skin/Aedes_aegypti_smashed_stabilityai_x4",
    "../Mosquito-on-human-skin/Aedes_aegypti_smashed_stabilityai_x4.csv",
    "aegypti",
)

annotate_images(
    "../Mosquito-on-human-skin/Aedes_aegypti_landing_stabilityai_x4",
    "../Mosquito-on-human-skin/Aedes_aegypti_landing_stabilityai_x4.csv",
    "aegypti",
)

annotate_images(
    "../mosqutio_kaggle/aegypti_0",
    "../mosqutio_kaggle/aegypti_0.csv",
    "aegypti",
)
