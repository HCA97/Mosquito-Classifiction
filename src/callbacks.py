import torch as th
import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from torchvision.utils import draw_bounding_boxes, make_grid



class LogAnnotations(Callback):
    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        print('Starting Log Annotations...')
        pl_module.eval()
        with th.no_grad():
            if pl_module.first_train_batch is not None:
                print('Drawing Train Anotations...')
                self.add_image(trainer, pl_module, pl_module.first_train_batch, 'train annotations')

            if pl_module.first_val_batch is not None:
                print('Drawing Val Annotations...')
                self.add_image(trainer, pl_module, pl_module.first_val_batch, 'val annotations')

        pl_module.train()


    def add_image(trainer: Trainer, pl_module: LightningModule, imgs: th.Tensor, text: str):
            bboxes, _ = pl_module(imgs)
            _, _, w, h = imgs.shape

            x_tl = min(w, bboxes[:, 0] * w)
            y_tl = min(h, bboxes[:, 1] * h)
            x_br = min(w, bboxes[:, 2] * w)
            y_br = min(h, bboxes[:, 3] * h)


            imgs_list = []
            for i in range(4):
                bbox = th.tensor([x_tl, y_tl, x_br, y_br])
                img = th.clone(imgs[i])
                img = (img - img.min()) / (img.max() - img.min())
                img_bbox = draw_bounding_boxes(th.tensor(255*img, dtype=th.uint8), th.unsqueeze(bbox, 0))
                imgs_list.append(img_bbox)

            grid = make_grid(imgs_list)

            trainer.logger.experiment.add_image(
                text, grid, global_step=trainer.current_epoch
            )
