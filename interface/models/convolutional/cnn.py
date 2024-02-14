from torch import cat, nn, Tensor
from models.model import NNModel


class CNNCombiExtraConvMoreLinear(nn.Module):
    initial_conv_out = 81
    second_conv_out = 729
    linear_in = 2187
    linear_hidden = 1800

    def __init__(self):
        super().__init__()

        self.three_by_three_9 = nn.Sequential(
            nn.Conv2d(9, self.initial_conv_out, kernel_size=3, stride=3),  # 3x3
            nn.ReLU(),
        )

        self.three_by_three_81 = nn.Sequential(
            nn.Conv2d(
                self.initial_conv_out, self.second_conv_out, kernel_size=3, stride=3
            ),  # 3x3
            nn.ReLU(),
        )

        self.one_by_nine = nn.Sequential(
            nn.Conv2d(9, self.initial_conv_out, kernel_size=(1, 9), stride=1),  # 9x1
            nn.ReLU(),
        )

        self.nine_by_one = nn.Sequential(
            nn.Conv2d(9, self.initial_conv_out, kernel_size=(9, 1), stride=1),  # 1x9
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(self.linear_in, self.linear_hidden),
            nn.ReLU(),
            nn.Linear(self.linear_hidden, self.linear_hidden),
            nn.ReLU(),
            nn.Linear(self.linear_hidden, self.linear_hidden),
            nn.ReLU(),
            nn.Linear(self.linear_hidden, 729),
        )

    def forward(self, x: Tensor) -> Tensor:
        mod_x = x.permute([0, 2, 1])
        mod_x = mod_x.unflatten(2, (9, 9))

        # conv 1
        conv1 = self.three_by_three_9(mod_x)
        conv1 = self.three_by_three_81(conv1).squeeze(2)

        # conv 2
        conv2 = self.one_by_nine(mod_x).squeeze(3).unflatten(2, (3, 3))
        conv2 = self.three_by_three_81(conv2).squeeze(2)

        # conv 3
        conv3 = self.nine_by_one(mod_x).squeeze(2).unflatten(2, (3, 3))
        conv3 = self.three_by_three_81(conv3).squeeze(2)

        conv_concat = cat((conv1, conv2, conv3), dim=2)
        conv_concat = conv_concat.flatten(1, 2)
        logits = self.linear(conv_concat)
        logits = logits.unflatten(1, (81, 9))
        return logits


cnn_model = NNModel(
    "./models/convolutional/CNNCombiExtraConvMoreLinear.model",
    CNNCombiExtraConvMoreLinear,
)
