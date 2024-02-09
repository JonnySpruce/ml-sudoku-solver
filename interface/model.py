from models.linear.linear_model import LinearNnXxl
import torch

model_name = "./models/linear/" + "LinearNnXxl.model.test"
model = LinearNnXxl().to("cpu")
model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))
