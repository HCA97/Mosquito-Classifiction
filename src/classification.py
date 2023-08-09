from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch as th
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torcheval.metrics.functional import multiclass_f1_score
from .models import CLIPClassifier


def accuracy(y1: th.Tensor, y2: th.Tensor):
    y1_argmax = y1.argmax(dim=1)
    y2_argmax = y2.argmax(dim=1)

    correct_sum = th.sum(y1_argmax == y2_argmax)
    return correct_sum / len(y1)


class MosquitoClassifier(pl.LightningModule):
    def __init__(
        self,
        n_classes: int = 6,
        model_name: str = "ViT-L-14",
        dataset: str = "datacomp_xl_s13b_b90k",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cls = CLIPClassifier(n_classes, model_name, dataset)
        self.scheduler = None
        self.n_classes = n_classes

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.cls(x)

    def lr_schedulers(self):
        # over-write this shit
        return self.scheduler

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(self.cls.get_learnable_params())
        self.scheduler = ReduceLROnPlateau(
            optimizer, "min", patience=2, factor=0.1, verbose=True
        )
        return optimizer

    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
        img, label_t = (
            train_batch["img"],
            train_batch["label"],
        )

        label_p = self.cls(img)
        label_loss = nn.CrossEntropyLoss()(label_p, label_t)

        self.log_dict(
            {
                "train_f1_score": multiclass_f1_score(
                    label_p,
                    label_t.argmax(dim=1),
                    num_classes=self.n_classes,
                ),
                "train_accuracy": accuracy(label_t, label_p),
                "train_loss": label_loss,
            }
        )

        return label_loss

    def validation_step(self, val_batch, batch_idx):
        img, label_t = (
            val_batch["img"],
            val_batch["label"],
        )

        label_p = self.cls(img)
        label_loss = nn.CrossEntropyLoss()(label_p, label_t)

        self.log_dict(
            {
                "val_f1_score": multiclass_f1_score(
                    label_p,
                    label_t.argmax(dim=1),
                    num_classes=self.n_classes,
                ),
                "val_accuracy": accuracy(label_t, label_p),
                "val_loss": label_loss,
            }
        )

    def on_epoch_end(self):
        if self.scheduler is not None:
            metrics = self.trainer.callback_metrics
            val_loss = metrics.get("val_loss")
            self.scheduler.step(val_loss)

        opt = self.optimizers(use_pl_optimizer=True)
        self.log("lr", opt.param_groups[0]["lr"])


if __name__ == "__main__":

    def test_accuracy():
        y1 = th.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        y2 = th.tensor([[0, 0, 0, 1], [0, 2, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0]])

        print(accuracy(y1, y2), 0.5)
