import torch.nn as nn


class TinyNerf(nn.Module):
    def __init__(self, input_size=6, output_size=4):
        super(TinyNerf, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.layer1(x)
        preds_color = y[..., :3]
        preds_density = y[..., 3:4]
        return preds_color, preds_density
