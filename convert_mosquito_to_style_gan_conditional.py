import os
import json

import cv2
import pandas as pd
from tqdm import tqdm

size = (256, 256)
img_dir = '../data/train'
annotation_csv = '../data/train.csv'


output_dir = '../data_style_gan_conditional'
labels = [
    "albopictus",
    "culex",
    "japonicus-koreicus",
    "culiseta",
    "anopheles",
    "aegypti",
]

def create_folders(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)



def create_style_gan_folder(df: pd.DataFrame, folder_name: str) -> list:
    def _loop(f_name: str, class_label: str, bbox: tuple) -> str:
        global img_dir, output_dir, size

        src_path = os.path.join(img_dir, f_name)
        
        dst_folder = os.path.join(output_dir, folder_name, class_label)
        os.makedirs(dst_folder, exist_ok=True)
        dst_path = os.path.join(dst_folder, f_name)

        # load image
        img = cv2.imread(src_path)
        
        # crop mosquito x: colums, y: rows
        img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img_res = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

        # save destination
        cv2.imwrite(dst_path, img_res)

        del img, img_res

    for i in tqdm(range(len(df))):
        f_name, _, _, x_tl, y_tl, x_br, y_br, class_label = df.iloc[i]
        _loop(f_name, class_label.replace('/', '-'), (x_tl, y_tl, x_br, y_br))

def create_dataset_json(folder_name: str):
    _path = os.path.join(output_dir, folder_name)

    dataset = {"labels": []}
    for root, dirs, files in os.walk(_path):
        if dirs:
            continue
        
        _dir = os.path.split(root)[-1]
        label_idx = labels.index(_dir)
        
        for file in files:
            dataset['labels'].append([os.path.join(_dir, file), label_idx])

    with open(os.path.join(_path, "dataset.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)

if __name__ == '__main__':
    df = pd.read_csv(annotation_csv)

    create_folders(output_dir)

    train_df = df.sample(frac=0.8, random_state=200)
    val_df = df.drop(train_df.index)

    create_style_gan_folder(train_df, 'train')
    create_dataset_json('train')
    # create_style_gan_folder(val_df, 'val')
        
        