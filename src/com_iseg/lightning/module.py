"""The lightning model defining forward pass and loss calculations."""

import torch
from torch import lgamma
from torch import nn
import lightning as L
from ..models.unet import UNet


def beta_nll_loss(alpha, beta, y):
    logB = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    return -((alpha - 1.0) * torch.log(y) + (beta - 1.0) * torch.log(1.0 - y) - logB)


def masked_loss(loss, mask):
    return loss[mask].mean()


class LaplaceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="none")

    def forward(self, y_hat, sigma, y):
        l1 = self.l1_loss(y_hat, y) / sigma
        log_sigma = torch.log(sigma)
        return l1 + log_sigma


class COMModule(L.LightningModule):
    def __init__(
        self, in_channels=1, ndim=3, depth: int = 3, num_channels_init: int = 64
    ):
        super().__init__()
        self.unet = UNet(
            conv_dims=ndim,
            in_channels=in_channels,
            ndim=ndim,
            depth=depth,
            num_channels_init=num_channels_init,
        )
        self.laplace_loss = LaplaceLoss()

    def predict_step(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        inputs, target_probs, target_coms, mask = batch
        mu, phi, com, sigma = self.unet(inputs)

        alpha = mu * phi
        beta = (1.0 - mu) * phi

        nll = beta_nll_loss(alpha, beta, target_probs)
        laplace_loss = self.laplace_loss(com, sigma, target_coms)

        loss = masked_loss(nll[:, 0] + laplace_loss.sum(axis=1), mask)
        self.log("loss", loss, prog_bar=True)
        for param in self.unet.parameters():
            if not torch.all(torch.isfinite(param)):
                self.log("all_isfinite", 0)
                break
        else:
            self.log("all_isfinite", 1)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.unet.parameters(), lr=0.1)
