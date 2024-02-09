from torch import nn, Tensor

class LinearNnXxl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (81, 9))

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(729, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 729),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return self.unflatten(logits)
