import torch.nn as nn


class TinyNerf(nn.Module):
    def __init__(self, input_size=5, output_size=3):
        super(TinyNerf, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        return x
