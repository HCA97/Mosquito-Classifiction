
import torch as th
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import open_clip


class LocalizationResNet50(nn.Module):
    def __init__(self, n_classes: int = 6, lrs: dict = {'hd_lr': 1e-3, 'hd_wd': 0.0, 'back_lrs': -1.0, 'back_wd': 0.0}) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            *list(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).children())[:-1]
        )

        self.bbox = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 4, bias=False)
        )
        self.label = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, n_classes)
        )
        self.lrs = lrs
        self.n_classes = n_classes


    def forward(self, x: th.tensor) -> tuple:        
        x = self.backbone(x)
        x = th.squeeze(x)
        return self.bbox(x), self.label(x)
    
    def get_learnable_params(self) -> list:

        back_lrs = self.lrs['back_lrs']
        back_wd = self.lrs['back_wd']
        hd_lr = self.lrs['hd_lr']
        hd_wd = self.lrs['hd_wd']
        
        parameters = []
        if back_lrs > 0 and back_wd >= 0:
            parameters.extend([
                {'params': p, "lr": back_lrs, "weight_decay": back_wd}
                for _, p in self.backbone.named_parameters()
            ])

        parameters.extend([
            {'params': p, "lr": hd_lr, "weight_decay": hd_wd}
            for _, p in self.bbox.named_parameters()
        ])

        parameters.extend([
            {'params': p, "lr": hd_lr, "weight_decay": hd_wd}
            for _, p in self.label.named_parameters()
        ])

        return parameters
        
class LocalizationCLIP(nn.Module):
    def __init__(self, n_classes: int = 6, lrs: dict = {'hd_lr': 1e-3, 'hd_wd': 0.0, 'back_lrs': -1.0, 'back_wd': 0.0}):
        super().__init__()
        self.backbone = open_clip.create_model_and_transforms('ViT-L-14', pretrained='datacomp_xl_s13b_b90k')[0].visual
        self.bbox = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 768),
            nn.LeakyReLU(),
            nn.Linear(768, 4, bias=False)
        )
        self.label = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 768),
            nn.LeakyReLU(),
            nn.Linear(768, n_classes),
        )

        self.n_classes = n_classes
        self.lrs = lrs

    def forward(self, x: th.tensor) -> th.tensor:        
        x = self.backbone(x)
        x = th.squeeze(x)
        return self.bbox(x), self.label(x)
    
    def get_parameter_section(self, parameters, lr=None, wd=None): 
        # https://github.com/IvanAer/G-Universal-CLIP
        parameter_settings = []


        lr_is_dict = isinstance(lr, dict)
        wd_is_dict = isinstance(wd, dict)

        layer_no = None
        for n,p in parameters:
            
            for split in n.split('.'):
                if split.isnumeric():
                    layer_no = int(split)
            
            if not layer_no:
                layer_no = 0
            
            if lr_is_dict:
                for k,v in lr.items():
                    if layer_no < int(k):
                        temp_lr = v
                        break
            else:
                temp_lr = lr

            if wd_is_dict:
                for k,v in wd.items():
                    if layer_no < int(k):
                        temp_wd = v
                        break
            else:
                temp_wd = wd

            parameter_setting = {"params" : p, "lr" : temp_lr, "weight_decay" : temp_wd}
            parameter_settings.append(parameter_setting)
        return parameter_settings

    def get_learnable_params(self) -> list:
        back_lrs = self.lrs['back_lrs']
        back_wd = self.lrs['back_wd']
        hd_lr = self.lrs['hd_lr']
        hd_wd = self.lrs['hd_wd']
        
        parameter_settings = [] 

        if back_lrs and back_wd:
            parameter_settings.extend(
                self.get_parameter_section(
                    [(n, p) for n, p in self.backbone.named_parameters()], 
                    lr=back_lrs, 
                    wd=back_wd
                )
            ) 

        parameter_settings.extend(
            self.get_parameter_section(
                [(n, p) for n, p in self.bbox.named_parameters()], 
                lr=hd_lr, 
                wd=hd_wd
            )
        ) 

        parameter_settings.extend(
            self.get_parameter_section(
                [(n, p) for n, p in self.label.named_parameters()], 
                lr=hd_lr, 
                wd=hd_wd
            )
        ) 

        return parameter_settings
    

if __name__ == '__main__':
    def test_model():
        model = LocalizationResNet50()

        x = th.rand([10, 3, 224, 224])

        print(model(x))

    def test_model2():
        model = LocalizationCLIP()
        x = th.rand([10, 3, 224, 224])

        print(model(x))

    test_model2()