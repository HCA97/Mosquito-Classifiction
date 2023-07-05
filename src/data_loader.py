import os
import logging
from typing import Optional

import torch as th
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms as T
import albumentations as A

logging.basicConfig(level=logging.INFO)


def normalize_locations(img_shape:tuple, bbox: tuple) -> tuple:
    x_tl, y_tl, x_br, y_br = bbox
    w, h = img_shape

    return [
        min(x_tl / w, 1.0),
        min(y_tl / h, 1.0),
        min(x_br / w, 1.0),
        min(y_br / h, 1.0)
    ]

def read_image_cv2(f_name: str, 
                   gray_scale: bool = False) -> np.ndarray:
    img = cv2.imread(f_name, cv2.IMREAD_ANYCOLOR if not gray_scale else cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class SimpleDetectionDataset(Dataset):
    def __init__(self, 
                 annotations_df : pd.DataFrame, 
                 img_dir: str, 
                 class_dict: dict, 
                 transform: Optional[T.Compose] = None,
                 data_augment: Optional[A.Compose] = None):
        self.df = annotations_df
        self.img_dir = img_dir
        self.class_dict = class_dict
        self.transform = transform
        self.data_augment = data_augment

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        cv2.setNumThreads(6)

        f_name, w, h, x_tl, y_tl, x_br, y_br, label = self.df.iloc[idx]

        img = read_image_cv2(os.path.join(self.img_dir, f_name))
        bbox_norm = normalize_locations((w, h), (x_tl, y_tl, x_br, y_br))

        
        if self.data_augment:
            transformed = self.data_augment(image=img, bboxes=[bbox_norm])
            img = transformed["image"]
            bbox_norm = transformed['bboxes'][0]

        if self.transform:
            img = self.transform(img)

        if self.class_dict:
            label = self.class_dict[label]
        return {
            "img": img, 
            "bbox_norm": th.tensor(bbox_norm, dtype=th.float32),
            "label": label
        }


if __name__ == '__main__':


    import torch as th
    import torchvision.transforms as T
    from torchvision.utils import draw_bounding_boxes
    import albumentations as A
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))

    


    img_dir = './data/train'
    annotations_csv = './data/train.csv'
    annotations_df = pd.read_csv(annotations_csv)
    class_dict = {
        "albopictus": th.tensor([1, 0, 0, 0, 0, 0], dtype=th.long),
        "culex": th.tensor([0, 1, 0, 0, 0, 0], dtype=th.long),
        "japonicus/koreicus": th.tensor([0, 0, 1, 0, 0, 0], dtype=th.long),
        "culiseta": th.tensor([0, 0, 0, 1, 0, 0], dtype=th.long),
        "anopheles": th.tensor([0, 0, 0, 0, 1, 0], dtype=th.long),
        "aegypti": th.tensor([0, 0, 0, 0, 0, 1], dtype=th.long),
    }



    transform = T.Compose([
        T.ToTensor()
    ])

    data_augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.MotionBlur(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        # A.Resize(224, 224, cv2.INTER_LINEAR),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=[]))

    ds = SimpleDetectionDataset(annotations_df=annotations_df, img_dir=img_dir, class_dict=class_dict, transform=transform, data_augment=data_augmentation)
    for i in range(10):
        res = ds[i]
        img = res['img']
        bbox = res["bbox_norm"]
        bbox[0] *= img.shape[2]
        bbox[2] *= img.shape[2]
        bbox[1] *= img.shape[1]
        bbox[3] *= img.shape[1]

        img_bbox = draw_bounding_boxes(th.tensor(255*img, dtype=th.uint8), th.unsqueeze(bbox, 0))
        show(img_bbox)
        plt.show()