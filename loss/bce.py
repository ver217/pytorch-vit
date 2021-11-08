import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassBCEWithLogitsLoss(nn.Module):
    """ BCE with optional one-hot from dense targets, label smoothing, thresholding
    """

    def __init__(self, smoothing=0.0):
        super().__init__()
        assert 0. <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        batch_size = x.size(0)
        if target.shape != x.shape:
            # NOTE currently assume smoothing or other label softening is applied upstream if targets are already sparse
            num_classes = x.shape[-1]
            # FIXME should off/on be different for smoothing w/ BCE? Other impl out there differ
            off_value = self.smoothing / num_classes
            on_value = 1. - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = torch.full(
                (batch_size, num_classes),
                off_value,
                device=x.device, dtype=x.dtype).scatter_(1, target, on_value)
        return F.binary_cross_entropy_with_logits(x, target, reduction='sum') / batch_size
