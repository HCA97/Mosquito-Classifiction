
from typing import Any, Iterator
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch as th
from torch import nn
import pytorch_lightning as pl
from torch.nn.parameter import Parameter
from torchvision.models import resnet50, ResNet50_Weights
import open_clip


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

class LocalizationResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            *list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-1]
        )

        self.mlp = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4, bias=False),
            nn.ReLU()
        )

    def forward(self, x: th.tensor) -> th.tensor:        
        x = self.backbone(x)
        x = th.squeeze(x)
        return self.mlp(x)
        
class LocalizationCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')[0].visual
        self.mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 4, bias=False),
            nn.ReLU()
        )
    def forward(self, x: th.tensor) -> th.tensor:        
        x = self.backbone(x)
        x = th.squeeze(x)
        return self.mlp(x)

class MosquitoLocalization(pl.LightningModule):
    def __init__(self, net_params: dict ={}, opt_params: dict={"lr": 1e-4}):
        super().__init__()

        self.detector = LocalizationCLIP(**net_params)

        self.net_params = net_params
        self.opt_params = opt_params


    def forward(self, x: th.tensor) -> th.tensor:
        return self.detector(x)
    
    def configure_optimizers(self):
        parameters_backbone = [
            # {'params': p, "lr": self.opt_params.get("lr", 1e-4) * 0, "weight_decay": self.opt_params.get("weight_decay", 1e-6) * 10}
            # for _, p in self.detector.backbone.named_parameters()
        ]
        parameters_mlp = [
            {'params': p, "lr": self.opt_params.get("lr", 1e-4), "weight_decay": self.opt_params.get("weight_decay", 1e-6)}
            for _, p in self.detector.mlp.named_parameters()
        ]
        optimizer = th.optim.Adam(parameters_mlp + parameters_backbone)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        x, y = train_batch["img"], train_batch["bbox_norm"]

        y_p = self.detector(x)
        loss = mse_loss(y, y_p)
        with th.no_grad():
            iou = iou_loss(y, y_p)

        self.log_dict({
            "train_loss": loss, 
            "train_iou_loss": th.mean(iou),
            "train_iou_0.75": th.sum(iou >= 0.75) / len(iou)
        })
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["img"], val_batch["bbox_norm"]
        y_p = self.detector(x)
        loss = mse_loss(y, y_p)
        iou = iou_loss(y, y_p)
        self.log_dict({
            "val_loss": loss, 
            "val_iou_loss": th.mean(iou),
            "val_iou_0.75": th.sum(iou >= 0.75) / len(iou)
        })


if __name__ == "__main__":

    def test_iou_loss():
        y1 = th.tensor([[0, 0, 10, 10]])
        y2 = th.tensor([[5, 5, 15, 15]])

        actual1 = iou_loss(y1, y2).item()
        actual2 = iou_loss(y2, y1).item()
        expected = 25 / 200

        print(actual1, actual2, expected)   


    def test_model():
        model = LocalizationNet()

        x = th.rand([10, 3, 224, 224])

        print(model(x))

    
