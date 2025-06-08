import torch.nn as nn


class NeRFLoss(nn.Module):
    def __init__(self):
        super(NeRFLoss, self).__init__()

    def forward(self, pred, gt):
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, gt)
        return loss
