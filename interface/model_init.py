from models.linear.linear_model import linear_model
from models.convolutional.cnn import cnn_model
import torch
from constants import LINEAR, CONVOLUTIONAL, DEVICE

models = {
    LINEAR: linear_model,
    CONVOLUTIONAL: cnn_model,
}


def instantiate_model(architecture: str) -> torch.nn.Module:
    nn_model = models[architecture]
    model = nn_model.model().to(DEVICE)
    model.load_state_dict(torch.load(nn_model.path, map_location=torch.device(DEVICE)))
    return model
