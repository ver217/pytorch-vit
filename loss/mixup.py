import torch.nn as nn


class MixupLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, inputs, targets):
        targets_a, targets_b, lam = targets
        return lam * self.loss_fn(inputs, targets_a) + (1 - lam) * self.loss_fn(inputs, targets_b)
