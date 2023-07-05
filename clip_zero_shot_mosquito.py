import os
from types import SimpleNamespace
from typing import Tuple

import open_clip
from open_clip import CLIP
from torchvision.transforms import Compose, ToPILImage

import faiss
import pandas as pd
import torch as th
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm

from src.data_loader import SimpleDetectionDataset


def load_clip_model(model_name:str, dataset:str) -> Tuple[CLIP, Compose]:
    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, 
                                                                        pretrained=dataset,
                                                                        precision='fp16')
    clip_model.eval()
    preprocess = Compose([ToPILImage(), preprocess])
    return clip_model, preprocess

class ZeroShot:
    def __init__(self, img_dir: str, df: pd.DataFrame, clip_params: SimpleNamespace):
        self.img_dir = img_dir
        self.df = df
        self.clip_params = clip_params
        self.device = 'cuda'

        self.faiss_index = None
        self.clip_model = None
        self.preprocess = None
        
    @th.no_grad()
    def index(self):
        self.clip_model, self.preprocess = load_clip_model(self.clip_params.model_name, self.clip_params.dataset)
        self.clip_model.to(self.device)

        dl = DataLoader(
            SimpleDetectionDataset(self.df, self.img_dir, None, self.preprocess), 
            batch_size=512, 
            shuffle=False, 
            num_workers=6, 
            persistent_workers=True, 
            pin_memory=True
        )
        
        embeddings = []
        for batch in tqdm(dl):
            x = batch['img']
                
            emb = self.clip_model.encode_image(x.half().to(self.device))
            emb = emb.data.cpu().numpy()
            emb = emb / np.linalg.norm(emb, axis=0)
            embeddings.append(emb)

        embeddings = np.concatenate(embeddings, axis=0)

        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)

    @th.no_grad()
    def search(self, img_path: str) -> str:
        label = ''
        dst = -1
        if self.faiss_index is not None:
            if self.clip_model is None or self.preprocess is None:
                self.clip_model, self.preprocess = load_clip_model(self.clip_params.model_name, self.clip_params.dataset)
                self.clip_model.to(self.device)

            # load img
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # preprocess
            x = self.preprocess(img)
            x = th.unsqueeze(x, 0)

            # get emb
            emb = self.clip_model.encode_image(x.half().to(self.device))
            emb = emb.detach().cpu().numpy()
            emb = emb / np.linalg.norm(emb)


            dst, ind = self.faiss_index.search(emb, 1)

            print(dst, ind)

            label = self.df.iloc[ind[0]].class_label
        
        return label, dst[0]
    

if __name__ == '__main__':
    img_dir = '../data/train'
    annotation_csv = '../data/train.csv'
    df = pd.read_csv(annotation_csv)
    clip_params = SimpleNamespace(model_name='ViT-B-32', dataset='laion2b_s34b_b79k')


    train_df = df.sample(frac=0.8, random_state=200)
    val_df = df.drop(train_df.index)


    zs = ZeroShot(img_dir, train_df, clip_params)
    zs.index()

    for f_name, label in val_df[['img_fName', 'class_label']]:
        img_path = os.path.join(img_dir, f_name)
        print('result', zs.search(img_path))
        print('label', label)
        print('--------------------')
    