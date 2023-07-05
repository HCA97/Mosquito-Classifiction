import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import pandas as pd
from tqdm import tqdm


img_dir = '../data/train'
annotation_csv = '../data/train.csv'


output_dir = '../data_funit'

labels = [
    "albopictus",
    "culex",
    "japonicus/koreicus",
    "culiseta",
    "anopheles",
    "aegypti",
]

def create_folders(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)


def create_funit_folder(df: pd.DataFrame, folder_name: str) -> list:
    def _loop(f_name: str, label: str, bbox: tuple) -> str:
        global img_dir, output_dir

        _label = label.replace('/', '_')

        src_path = os.path.join(img_dir, f_name)
        
        dst_folder = os.path.join(output_dir, folder_name, _label)
        os.makedirs(dst_folder, exist_ok=True)
        dst_path = os.path.join(dst_folder, f_name)

        # load image
        img = cv2.imread(src_path)
        
        # crop mosquito x: colums, y: rows
        img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # save destination
        cv2.imwrite(dst_path, img)

        return os.path.join(_label, f_name)

    res = []
    with ThreadPoolExecutor(10) as exe:
        jobs = []
        for label in labels:
            df_label = df[df.class_label == label]
            for i in range(len(df_label)):
                f_name, _, _, x_tl, y_tl, x_br, y_br, label = df_label.iloc[i]
                jobs.append(exe.submit(_loop, f_name, label, (x_tl, y_tl, x_br, y_br)))

        res = [job.result() for job in tqdm(jobs)]

    return res
            


if __name__ == '__main__':
    df = pd.read_csv(annotation_csv)

    create_folders(output_dir)

    train_df = df.sample(frac=0.8, random_state=200)
    val_df = df.drop(train_df.index)

    res_train = create_funit_folder(train_df, 'train')
    with open('mosquitos_list_train.txt', 'w') as f:
        for line in res_train:
            f.write(f'{line}\n')

    res_val = create_funit_folder(val_df, 'val')
    with open('mosquitos_list_val.txt', 'w') as f:
        for line in res_val:
            f.write(f'{line}\n')

        
        