from copy import deepcopy

from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch as th
from torch import nn
import pytorch_lightning as pl
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from transformers import get_linear_schedule_with_warmup

from .models import CLIPClassifier
from .convnext_meta import build_covnext


def f1(y_true: th.Tensor, y_pred: th.Tensor):
    y_pred = th.round(y_pred)
    tp = th.sum((y_true * y_pred).float(), dim=0)
    tn = th.sum(((1 - y_true) * (1 - y_pred)).float(), dim=0)
    fp = th.sum(((1 - y_true) * y_pred).float(), dim=0)
    fn = th.sum((y_true * (1 - y_pred)).float(), dim=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = th.where(th.isnan(f1), th.zeros_like(f1), f1)
    return th.mean(f1)


def f1_loss(y_true: th.Tensor, y_pred: th.Tensor):
    tp = th.sum((y_true * y_pred).float(), dim=0)
    tn = th.sum(((1 - y_true) * (1 - y_pred)).float(), dim=0)
    fp = th.sum(((1 - y_true) * y_pred).float(), dim=0)
    fn = th.sum((y_true * (1 - y_pred)).float(), dim=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = th.where(th.isnan(f1), th.zeros_like(f1), f1)
    return 1 - th.mean(f1)


def accuracy(y1: th.Tensor, y2: th.Tensor):
    y1_argmax = y1.argmax(dim=1)
    y2_argmax = y2.argmax(dim=1)

    correct_sum = th.sum(y1_argmax == y2_argmax)
    return correct_sum / len(y1)


class EMA(nn.Module):
    """Model Exponential Moving Average V2 from timm"""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model: nn.Module, update_fn):
        with th.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class MosquitoClassifier(pl.LightningModule):
    def __init__(
        self,
        n_classes: int = 6,
        model_name: str = "ViT-L-14",
        dataset: str = "datacomp_xl_s13b_b90k",
        freeze_backbones: bool = False,
        head_version: int = 0,
        warm_up_steps: int = 2000,
        bs: int = 64,
        data_aug: str = "",
        loss_func: str = "ce",
        epochs: int = 5,
        label_smoothing: float = 0.0,
        hd_lr: float = 3e-4,
        hd_wd: float = 1e-5,
        img_size: tuple = (224, 224),
        use_ema: bool = False,
        use_same_split_as_yolo: bool = False,
        shift_box: bool = False,
        max_steps: int = 12400,
    ):
        super().__init__()
        self.save_hyperparameters()

        if dataset == "imagenet":
            self.cls = build_covnext(
                model_name, n_classes, hd_lr, hd_wd, freeze_backbones
            )

        else:
            self.cls = CLIPClassifier(
                n_classes, model_name, dataset, head_version, hd_lr, hd_wd
            )
            if freeze_backbones:
                for param in self.cls.backbone.parameters():
                    param.requires_grad = False

        self.use_ema = use_ema
        if self.use_ema:
            self.ema = EMA(self.cls, decay=0.995)

        self.scheduler = None
        self.n_classes = n_classes
        self.warm_up_steps = warm_up_steps
        self.loss_func = loss_func
        self.label_smoothing = label_smoothing
        self.max_steps = max_steps

        self.val_labels_t = []
        self.val_labels_p = []

        self.train_labels_t = []
        self.train_labels_p = []

    def on_before_backward(self, *args, **kwargs):
        # ref: https://github.com/Lightning-AI/lightning/issues/10914
        if self.use_ema:
            self.ema.update(self.cls)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.use_ema and not self.training:
            return self.ema.module(x)
        return self.cls(x)

    def lr_schedulers(self):
        # over-write this shit
        return self.scheduler

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.cls.get_learnable_params())
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warm_up_steps,
            num_training_steps=self.max_steps,  # not sure what to set
        )
        return optimizer

    def compute_loss(self, label_t: th.Tensor, label_p: th.Tensor) -> th.Tensor:
        if self.loss_func == "f1":
            label_loss = f1_loss(label_t, th.nn.functional.softmax(label_p, dim=1))
        elif self.loss_func == "ce+f1":
            label_loss = f1_loss(
                label_t, th.nn.functional.softmax(label_p, dim=1)
            ) + nn.CrossEntropyLoss()(label_p, label_t)
        else:
            label_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(
                label_p, label_t
            )

        return label_loss

    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        img, label_t = (
            train_batch["img"],
            train_batch["label"],
        )

        label_p = self.cls(img)
        label_loss = self.compute_loss(label_t, label_p)

        self.train_labels_t.append(label_t.detach().cpu())
        self.train_labels_p.append(label_p.detach().cpu())

        self.log("train_loss", label_loss)

        if self.scheduler is not None:
            self.scheduler.step()

        return label_loss

    def on_train_epoch_end(self) -> None:
        label_p = th.concatenate(self.train_labels_p)
        label_t = th.concatenate(self.train_labels_t)

        self.log_dict(
            {
                "train_f1_score": multiclass_f1_score(
                    label_p,
                    label_t.argmax(dim=1),
                    num_classes=self.n_classes,
                    average="macro",
                ),
                "train_multiclass_accuracy": multiclass_accuracy(
                    label_p,
                    label_t.argmax(dim=1),
                    num_classes=self.n_classes,
                    average="macro",
                ),
                "train_accuracy": accuracy(label_t, label_p),
            }
        )

        self.train_labels_t = []
        self.train_labels_p = []

    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT:
        img, label_t = (
            val_batch["img"],
            val_batch["label"],
        )
        if self.use_ema:
            label_p = self.ema.module(img)
        else:
            label_p = self.cls(img)

        label_loss = self.compute_loss(label_t, label_p)

        self.val_labels_t.append(label_t.detach().cpu())
        self.val_labels_p.append(label_p.detach().cpu())

        self.log("val_loss", label_loss)

        return label_loss

    def on_validation_epoch_end(self):
        label_p = th.concatenate(self.val_labels_p)
        label_t = th.concatenate(self.val_labels_t)

        self.log_dict(
            {
                "val_f1_score": multiclass_f1_score(
                    label_p,
                    label_t.argmax(dim=1),
                    num_classes=self.n_classes,
                    average="macro",
                ),
                "val_multiclass_accuracy": multiclass_accuracy(
                    label_p,
                    label_t.argmax(dim=1),
                    num_classes=self.n_classes,
                    average="macro",
                ),
                "val_accuracy": accuracy(label_t, label_p),
            }
        )

        self.val_labels_t = []
        self.val_labels_p = []

    def on_epoch_end(self):
        opt = self.optimizers(use_pl_optimizer=True)
        self.log("lr", opt.param_groups[0]["lr"])


if __name__ == "__main__":

    def test_accuracy():
        y1 = th.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        y2 = th.tensor([[0, 0, 0, 1], [0, 2, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0]])

        print(f1_loss(y1, y2), 0.5)

    test_accuracy()
