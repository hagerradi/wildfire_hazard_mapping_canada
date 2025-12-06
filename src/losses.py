# Defintions of loss functions
import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # internally handles sigmoid
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        bce_loss = self.bce(logits, targets)

        if mask is not None:
            bce_loss = bce_loss * mask
            return bce_loss.sum() / mask.sum()

        return bce_loss.mean()


# TODO: remove later
if __name__ == "__main__":
    preds = torch.randn(2, 1, 256, 256)
    y = torch.ones(2, 1, 256, 256)
    loss_fn = BCELoss()
    output = loss_fn(preds, y)
    print(f"Loss: {output.item()}")
