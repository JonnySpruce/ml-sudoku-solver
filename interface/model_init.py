from models.linear.linear_model import linear_model
from models.convolutional.cnn import cnn_model
import torch
from constants import LINEAR, CONVOLUTIONAL

models = {
    LINEAR: linear_model,
    CONVOLUTIONAL: cnn_model,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def instantiate_model(architecture: str) -> torch.nn.Module:
    nn_model = models[architecture]
    model = nn_model.model().to(device)
    model.load_state_dict(torch.load(nn_model.path, map_location=torch.device(device)))
    return model
