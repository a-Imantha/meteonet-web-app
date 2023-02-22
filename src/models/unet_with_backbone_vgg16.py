import torch
import segmentation_models_pytorch as smp

MODEL_PATH_VGG16 = "src/models/state_dicts/model-vgg16-1-state-dict.pt"


def get_model():
    model = smp.Unet('vgg16', in_channels=11, classes=1, encoder_weights=None)
    model.load_state_dict(torch.load(MODEL_PATH_VGG16, map_location=torch.device('cpu')))

    return model
