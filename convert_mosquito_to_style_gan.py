import os
from multiprocessing import Pool

import cv2
import pandas as pd
from tqdm import tqdm

size = (256, 256)
img_dir = '../data/train'
annotation_csv = '../data/train.csv'


output_dir = '../data_style_gan'


def create_folders(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)



def create_style_gan_folder(df: pd.DataFrame, folder_name: str) -> list:
    def _loop(f_name: str, bbox: tuple) -> str:
        global img_dir, output_dir, size

        src_path = os.path.join(img_dir, f_name)
        
        dst_folder = os.path.join(output_dir, folder_name)
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
        f_name, _, _, x_tl, y_tl, x_br, y_br, _ = df.iloc[i]
        _loop(f_name, (x_tl, y_tl, x_br, y_br))


if __name__ == '__main__':
    df = pd.read_csv(annotation_csv)

    create_folders(output_dir)

    train_df = df.sample(frac=0.8, random_state=200)
    val_df = df.drop(train_df.index)

    create_style_gan_folder(train_df, 'train')
    create_data_json('train')
    create_style_gan_folder(val_df, 'val')
        
        