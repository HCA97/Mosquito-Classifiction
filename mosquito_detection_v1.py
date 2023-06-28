from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd
import torch as th 
import pytorch_lightning as pl
from tqdm import tqdm

import src.localization as lc
import src.data_loader as dl



img_dir = './data/train'
annotations_csv = './data/train.csv'
annotations_df = pd.read_csv(annotations_csv)

transform = T.Compose([
    T.ToTensor(), 
    T.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225)
    )
])

class_dict = {
    "albopictus": th.tensor([1, 0, 0, 0, 0, 0]),
    "culex": th.tensor([0, 1, 0, 0, 0, 0]),
    "japonicus/koreicus": th.tensor([0, 0, 1, 0, 0, 0]),
    "culiseta": th.tensor([0, 0, 0, 1, 0, 0]),
    "anopheles": th.tensor([0, 0, 0, 0, 1, 0]),
    "aegypti": th.tensor([0, 0, 0, 0, 0, 1]),
}

train_df = annotations_df.sample(frac=0.95,random_state=200)
val_df = annotations_df.drop(train_df.index)

train_dataset = dl.SimpleDetectionDataset(train_df, img_dir, class_dict, transform)
train_dataloader = DataLoader(train_dataset, 
                              batch_size=64, 
                              shuffle=True, 
                              num_workers=12, 
                              persistent_workers=True, 
                              pin_memory=True)

val_dataset = dl.SimpleDetectionDataset(val_df, img_dir, class_dict, transform)
val_dataloader = DataLoader(val_dataset, 
                         batch_size=64, 
                          shuffle=False, 
                          num_workers=12, 
                          persistent_workers=True, 
                          pin_memory=True)


# dl.fill_up_cache(val_df, img_dir)
# print("CACHE SIZE", len(dl.CACHE))

# for batch in tqdm(val_dataloader):
#     pass

# # print(len(train_dataset.cache))

# for batch in tqdm(val_dataloader):
#     pass


th.set_float32_matmul_precision('high')
model = lc.MosquitoLocalization()
trainer = pl.Trainer(accelerator="gpu", 
                     precision='16-mixed',
                     max_epochs=30, 
                     logger=True)

trainer.fit(model=model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader)