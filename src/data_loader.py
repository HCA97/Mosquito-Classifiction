import os
import logging

from tqdm import tqdm
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)


def normalize_locations(img_shape:tuple, bbox: tuple) -> tuple:
    x_tl, y_tl, x_br, y_br = bbox
    w, h = img_shape

    return [
        x_tl / w,
        y_tl / h,
        x_br / w,
        y_br / h
    ]

def fill_up_cache(df: pd.DataFrame, img_dir: str):
    for i in tqdm(range(len(df))):
        f_name = os.path.join(img_dir, df.iloc[i, 0])
        read_image_cv2(f_name)

def crate_mask(size: tuple, bbox: tuple) -> np.ndarray:
    mask = np.zeros(size, dtype=np.uint8)
    x_tl, y_tl, x_br, y_br = bbox
    mask[y_tl:y_br, x_tl:x_br] = 255
    return mask

def get_bbox(mask: np.ndarray) -> tuple:
    y, x = np.where(mask > 0)
    return (
        x.min(),
        y.min(),
        x.max(),
        y.max()
    )

def read_image_cv2(f_name: str, 
                   gray_scale: bool = False) -> np.ndarray:
    img = cv2.imread(f_name, cv2.IMREAD_ANYCOLOR if not gray_scale else cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class SimpleDetectionDataset(Dataset):
    def __init__(self, annotations_df : pd.DataFrame, img_dir: str, class_dict: dict, transform =None, data_augment=None):
        self.df = annotations_df
        self.img_dir = img_dir
        self.transform = transform
        self.class_dict = class_dict
        self.data_augment = data_augment

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        # cv2.setNumThreads(6)
        logging.info(f'processing {idx}')

        f_name, w, h, x_tl, y_tl, x_br, y_br, label = self.df.iloc[idx]

        img = read_image_cv2(os.path.join(self.img_dir, f_name))
        bbox_norm = normalize_locations((w, h), (x_tl, y_tl, x_br, y_br))

        
        if self.data_augment:
            mask = crate_mask((h, w), (x_tl, y_tl, x_br, y_br))

            # import matplotlib.pyplot as plt
            # plt.subplot(1, 2, 2)
            # plt.imshow(mask)
            # plt.subplot(1, 2, 1)
            # plt.imshow(img)
            # plt.show()
            transformed = self.data_augment(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

            # import matplotlib.pyplot as plt
            # plt.subplot(1, 2, 2)
            # plt.imshow(mask)
            # plt.subplot(1, 2, 1)
            # plt.imshow(img)
            # plt.show()

            bbox = get_bbox(mask)
            print(f'{bbox} - {img.shape[:2]} - {f_name}')
            h, w, _ = img.shape
            bbox_norm = normalize_locations((w, h), bbox)
        
        if self.transform:
            img = self.transform(img)

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
            # axs[0, i].set(exticklabels=[], yticklabels=[], xticks=[], yticks=[])

    


    img_dir = './data/train'
    annotations_csv = './data/train.csv'
    annotations_df = pd.read_csv(annotations_csv)
    class_dict = {
        "albopictus": th.tensor([1, 0, 0, 0, 0, 0]),
        "culex": th.tensor([0, 1, 0, 0, 0, 0]),
        "japonicus/koreicus": th.tensor([0, 0, 1, 0, 0, 0]),
        "culiseta": th.tensor([0, 0, 0, 1, 0, 0]),
        "anopheles": th.tensor([0, 0, 0, 0, 1, 0]),
        "aegypti": th.tensor([0, 0, 0, 0, 0, 1]),
    }


    transform = T.Compose([
        T.ToTensor()
    ])

    data_augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomScale(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Resize(224, 224, cv2.INTER_LINEAR),
    ])

    ds = SimpleDetectionDataset(annotations_df=annotations_df, img_dir=img_dir, class_dict=class_dict, transform=transform, data_augment=data_augmentation)
    i = 0
    res = ds[i]
    img = res['img']
    bbox = res["bbox_norm"]
    bbox[0] *= img.shape[1]
    bbox[2] *= img.shape[1]
    bbox[1] *= img.shape[2]
    bbox[3] *= img.shape[2]

    img_bbox = draw_bounding_boxes(th.tensor(255*img, dtype=th.uint8), th.unsqueeze(bbox, 0))
    show(img_bbox)
    plt.show()