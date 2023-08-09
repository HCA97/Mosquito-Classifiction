import torch as th
from torch import nn
import open_clip


class CLIPClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int = 6,
        model_name: str = "ViT-L-14",
        data: str = "datacomp_xl_s13b_b90k",
    ):
        super().__init__()
        self.backbone = open_clip.create_model_and_transforms(
            model_name, pretrained=data
        )[0].visual

        if model_name == "ViT-L-14":
            self.n = 768
            self.lrs = dict(
                back_lrs={"8": 1.25e-6, "16": 2.5e-6, "20": 5e-6, "24": 10e-6},
                back_wd=1e-3,
                hd_lr=3e-4,
                hd_wd=1e-5,
            )
        elif model_name == "ViT-H-14":
            self.n = 1024
            self.lrs = {
                "back_lrs": {"10": 1.25e-6, "20": 2.5e-6, "26": 5e-6, "32": 10e-6},
                "back_wd": 1e-3,
                "hd_lr": 3e-4,
                "hd_wd": 1e-5,
            }
        elif model_name == "ViT-B-16":
            self.n = 512
            self.lrs = {
                "back_lrs": {"1": 2.5e-6, "7": 5e-6, "12": 10e-6},
                "back_wd": 1e-3,
                "hd_lr": 3e-4,
                "hd_wd": 1e-5,
            }
        else:
            raise ValueError

        self.label = nn.Sequential(
            nn.Dropout1d(),
            nn.LeakyReLU(),
            nn.Linear(self.n, n_classes),
        )

        self.n_classes = n_classes

    def forward(self, x: th.tensor) -> th.tensor:
        x = self.backbone(x)
        x = th.squeeze(x)
        return self.label(x)

    def get_parameter_section(self, parameters, lr=None, wd=None):
        # https://github.com/IvanAer/G-Universal-CLIP
        parameter_settings = []

        lr_is_dict = isinstance(lr, dict)
        wd_is_dict = isinstance(wd, dict)

        layer_no = None
        for n, p in parameters:
            for split in n.split("."):
                if split.isnumeric():
                    layer_no = int(split)

            if not layer_no:
                layer_no = 0

            if lr_is_dict:
                for k, v in lr.items():
                    if layer_no < int(k):
                        temp_lr = v
                        break
            else:
                temp_lr = lr

            if wd_is_dict:
                for k, v in wd.items():
                    if layer_no < int(k):
                        temp_wd = v
                        break
            else:
                temp_wd = wd

            parameter_setting = {"params": p, "lr": temp_lr, "weight_decay": temp_wd}
            parameter_settings.append(parameter_setting)
        return parameter_settings

    def get_learnable_params(self) -> list:
        back_lrs = self.lrs["back_lrs"]
        back_wd = self.lrs["back_wd"]
        hd_lr = self.lrs["hd_lr"]
        hd_wd = self.lrs["hd_wd"]

        parameter_settings = []

        if back_lrs and back_wd:
            parameter_settings.extend(
                self.get_parameter_section(
                    [(n, p) for n, p in self.backbone.named_parameters()],
                    lr=back_lrs,
                    wd=back_wd,
                )
            )

        parameter_settings.extend(
            self.get_parameter_section(
                [(n, p) for n, p in self.label.named_parameters()], lr=hd_lr, wd=hd_wd
            )
        )

        return parameter_settings


if __name__ == "__main__":
    m = CLIPClassifier(6, "ViT-B-16", "datacomp_l_s1b_b8k")

    x = th.rand([10, 3, 224, 224])

    print(m(x).shape)
