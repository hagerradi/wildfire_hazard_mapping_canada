# Defintions of loss functions
import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.tensor, targets: torch.tensor):
        # internally handles sigmoid
        bce_loss = self.bce(logits, targets)

        return bce_loss
    
# TODO: remove later
if __name__ == "__main__":
    preds = torch.randn(2, 1, 256, 256)
    y = torch.ones(2, 1, 256, 256)
    loss_fn = BCELoss()
    output = loss_fn(preds, y)
    print(f"Loss: {output.item()}")
