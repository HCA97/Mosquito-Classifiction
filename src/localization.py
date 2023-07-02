
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch as th
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loc_models import LocalizationResNet50, LocalizationCLIP

def iou_loss(y1, y2):
    # B, 4
    # B, [x_tl, y_tl, x_br, y_br]
    # x_tl <= x_br, y_tl <= y_br

    # intersect
    intersect = th.abs(th.minimum(y1[:, 2], y2[:, 2]) - th.maximum(y1[:, 0], y2[:, 0])) * \
                th.abs(th.minimum(y1[:, 3], y2[:, 3]) - th.maximum(y1[:, 1], y2[:, 1])) 

    # unioun
    union = th.abs(y1[:, 2] - y1[:, 0]) * th.abs(y1[:, 3] - y1[:, 1]) + \
            th.abs(y2[:, 2] - y2[:, 0]) * th.abs(y2[:, 3] - y2[:, 1])

    # result
    return intersect / (union + 1e-10)


def mse_loss(y1, y2):
    return nn.functional.mse_loss(y1, y2)


def accuracy(y1: th.Tensor, y2: th.Tensor):
    y1_argmax = y1.argmax(dim=1)
    y2_argmax = y2.argmax(dim=1)

    correct_sum = th.sum(y1_argmax == y2_argmax)
    return correct_sum / len(y1)


class MosquitoLocalization(pl.LightningModule):
    def __init__(self, net_name = 'resnet50', net_params: dict ={}, freeze_backbones: bool =True):
        super().__init__()
        self.save_hyperparameters()


        if net_name == 'resnet50':
            self.detector = LocalizationResNet50(**net_params)          
        else:
            self.detector = LocalizationCLIP(**net_params)

        if freeze_backbones:
            self.freeze_back_bone()

        self.net_params = net_params

        self.scheduler = None
        self.first_train_batch: th.Tensor = None
        self.first_val_batch: th.Tensor = None

    def freeze_back_bone(self):
        for param in self.detector.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: th.Tensor) -> tuple:
        return self.detector(x)
    
    def lr_schedulers(self):
        # over-write this shit
        return self.scheduler
    
    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.detector.get_learnable_params())
        self.scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        img, bbox_t, label_t = train_batch["img"], train_batch["bbox_norm"], train_batch['label']

        if self.first_train_batch is None:
            self.first_train_batch = th.clone(img.detach())

        bbox_p, label_p = self.detector(img)
        bbox_loss = mse_loss(bbox_t, bbox_p)
        bbox_iou = iou_loss(bbox_t, bbox_p)
        label_loss = nn.CrossEntropyLoss()(label_p, label_t)

        self.log_dict({
            "train_loss": bbox_loss + label_loss, 
            "train_iou_loss": th.mean(bbox_iou),
            "train_iou_0.75": th.sum(bbox_iou >= 0.75) / len(bbox_iou),
            "train_accuracy": accuracy(label_t, label_p),
            "train_label_loss": label_loss,
            "train_bbox_loss": bbox_loss
        })
        
        return bbox_loss + label_loss

    def validation_step(self, val_batch, batch_idx):
        img, bbox_t, label_t = val_batch["img"], val_batch["bbox_norm"], val_batch['label']

        if self.first_val_batch is None:
            self.first_val_batch = th.clone(img.detach())

        bbox_p, label_p = self.detector(img)
        bbox_loss = mse_loss(bbox_t, bbox_p)
        bbox_iou = iou_loss(bbox_t, bbox_p)
        label_loss = nn.CrossEntropyLoss()(label_p, label_t)

        self.log_dict({
            "val_loss": bbox_loss + label_loss, 
            "val_iou_loss": th.mean(bbox_iou),
            "val_iou_0.75": th.sum(bbox_iou >= 0.75) / len(bbox_iou),
            "val_accuracy": accuracy(label_t, label_p),
            "val_label_loss": label_loss,
            'val_bbox_loss': bbox_loss
        })

    def on_epoch_end(self):
        if self.scheduler is not None:
            metrics = self.trainer.callback_metrics
            val_loss = metrics.get('val_loss')
            self.scheduler.step(val_loss)
            
        opt = self.optimizers(use_pl_optimizer=True)
        self.log("lr", opt.param_groups[0]["lr"])


if __name__ == "__main__":

    def test_iou_loss():
        y1 = th.tensor([[0, 0, 10, 10]])
        y2 = th.tensor([[5, 5, 15, 15]])

        actual1 = iou_loss(y1, y2).item()
        actual2 = iou_loss(y2, y1).item()
        expected = 25 / 200

        print(actual1, actual2, expected)   

    def test_accuracy():
        y1 = th.tensor([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]])
        
        y2 = th.tensor([[0, 0, 0, 1],
                        [0, 2, 0, 0],
                        [0, 2, 0, 0],
                        [1, 0, 0, 0]])


        print(accuracy(y1, y2), 0.5)