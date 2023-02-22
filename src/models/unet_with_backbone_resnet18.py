import torch
import segmentation_models_pytorch as smp

MODEL_PATH_RESNET18 = "src/models/state_dicts/model-resnet18-1-state-dict.pt"

def get_model():
    model = smp.Unet('resnet18', in_channels=11, classes=1, encoder_weights=None)
    model.load_state_dict(torch.load(MODEL_PATH_RESNET18, map_location=torch.device('cpu')))

    return model
