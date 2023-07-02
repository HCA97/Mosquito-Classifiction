from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd
import torch as th 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import albumentations as A

import src.localization as lc
import src.data_loader as dl
from src.callbacks import LogAnnotations



img_dir = '../data/train'
annotations_csv = '../data/train.csv'
annotations_df = pd.read_csv(annotations_csv)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(size=(1024, 1024), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(), 
    T.Normalize(
        mean=(0.485, 0.456, 0.406), 
        std=(0.229, 0.224, 0.225)
    )
])

data_augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.MotionBlur(p=0.5),
        A.ShiftScaleRotate(p=0.5),
], bbox_params=A.BboxParams(format='albumentations', label_fields=[]))

class_dict = {
    "albopictus": th.tensor([1, 0, 0, 0, 0, 0], dtype=th.float),
    "culex": th.tensor([0, 1, 0, 0, 0, 0],dtype=th.float),
    "japonicus/koreicus": th.tensor([0, 0, 1, 0, 0, 0], dtype=th.float),
    "culiseta": th.tensor([0, 0, 0, 1, 0, 0], dtype=th.float),
    "anopheles": th.tensor([0, 0, 0, 0, 1, 0], dtype=th.float),
    "aegypti": th.tensor([0, 0, 0, 0, 0, 1], dtype=th.float),
}

# class_dict = {
#     "albopictus": 0,
#     "culex": 1,
#     "japonicus/koreicus": 2,
#     "culiseta": 3,
#     "anopheles": 4,
#     "aegypti": 5,
# }


train_df = annotations_df.sample(frac=0.8, random_state=200)
val_df = annotations_df.drop(train_df.index)

train_dataset = dl.SimpleDetectionDataset(train_df, img_dir, class_dict, transform, data_augmentation)
train_dataloader = DataLoader(train_dataset, 
                              batch_size=64, 
                              shuffle=True, 
                              num_workers=6, 
                              drop_last=True,
                              persistent_workers=True, 
                              pin_memory=True)

val_dataset = dl.SimpleDetectionDataset(val_df, img_dir, class_dict, transform)
val_dataloader = DataLoader(val_dataset, 
                         batch_size=64, 
                          shuffle=False, 
                          num_workers=6, 
                          persistent_workers=True, 
                          pin_memory=True)

callbacks = [
        ModelCheckpoint(every_n_epochs=1, save_top_k=-1, filename="{epoch}-{val_loss}"),
        EarlyStopping(monitor="val_loss"),
        LogAnnotations()
]

th.set_float32_matmul_precision('high')
model = lc.MosquitoLocalization(net_name='resnet50')
trainer = pl.Trainer(accelerator="gpu", 
                     precision='16-mixed',
                     max_epochs=30, 
                     logger=True, 
                     callbacks=callbacks)

trainer.fit(model=model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader)