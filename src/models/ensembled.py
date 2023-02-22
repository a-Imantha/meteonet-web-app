import torch.nn as nn
import torch
import unet_with_backbone_vgg16 as vgg16_unet
import unet_with_backbone_resnet18 as resnet18_unet

MODEL_PATH_ENSEMBLED = "./state_dicts/model-ensembled-state-dict.pt"


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()

        self.models = models
        self.ens_model = nn.Sequential(
            nn.Conv2d(len(models), 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        with torch.no_grad():
            x_ = torch.cat((self.models[0](x),), dim=1)
            for model in self.models[1:]:
                x_ = torch.cat((x_, model(x)), dim=1)

        out = self.ens_model(x_)
        return out


def get_model():
    models = [vgg16_unet.get_model(), resnet18_unet.get_mode()]

    ens = EnsembleModel([model for model in models])
    ens.load_state_dict(torch.load(MODEL_PATH_ENSEMBLED, map_location=torch.device('cpu')))
    return ens
