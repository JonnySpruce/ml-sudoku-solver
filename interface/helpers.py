from torch import Tensor
import torch
from best_guess import best_predict
from check import check_sudoku
from constants import IMMEDIATE
from model_init import instantiate_model
from gradio import Info
import numpy as np


def chunks(lst: list[int], n: int) -> list[list[int]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def convert_one_hot(one_hot_tensor: Tensor) -> list[list[int]]:
    arg_max = torch.argmax(one_hot_tensor, dim=2) + 1
    ans = chunks(arg_max.tolist()[0], 9)
    Info("Correct" if check_sudoku(ans) else "Wrong")
    return ans


def predict_string(
    sudoku: str, architecture: str, prediction_type: str
) -> list[list[int]]:
    sudoku = [int(num) for num in sudoku]
    sudoku = torch.nn.functional.one_hot(torch.from_numpy(np.array(sudoku)), 10)[
        :, 1:
    ].type(torch.float)
    model = instantiate_model(architecture)
    inp = sudoku.unsqueeze(0)
    result = (
        model(inp)
        if prediction_type == IMMEDIATE
        else best_predict(inp, model)
    )
    return convert_one_hot(result)
