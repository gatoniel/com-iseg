"""The lightning model defining forward pass and loss calculations."""

import torch
from torch import lgamma
from torch import nn
import lightning as L
from ..models.unet import UNet


def beta_nll_loss(alpha, beta, y, eps=1e-6):
    logB = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    if not torch.all(torch.isfinite(logB)):
        print("non finite value found in logB")
        raise ValueError
    loss = (alpha - 1.0) * torch.log(y) + (beta - 1.0) * torch.log(1.0 - y)
    if not torch.all(torch.isfinite(loss)):
        print("non finite value found in beta_nll_loss loss")
        raise ValueError
    return -(loss - logB)


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
        results = self.unet(inputs)
        alpha, beta, com, sigma = results
        for i, res in enumerate(results):
            if not torch.all(torch.isfinite(res)):
                print("non finite value found", i)
                raise ValueError

        # alpha = mu * phi
        # beta = (1.0 - mu) * phi

        nll = masked_loss(beta_nll_loss(alpha, beta, target_probs)[:, 0], mask)
        if not torch.all(torch.isfinite(nll)):
            print("non finite value found in nll")
            raise ValueError
        laplace_loss = masked_loss(
            self.laplace_loss(com, sigma, target_coms).sum(axis=1), mask
        )
        if not torch.all(torch.isfinite(laplace_loss)):
            print("non finite value found in laplace_loss")
            raise ValueError

        loss = nll + 2 * laplace_loss
        self.log("loss", loss, prog_bar=True)
        self.log("nll_loss", nll, prog_bar=True)
        self.log("laplace_loss", laplace_loss, prog_bar=True)
        for param in self.unet.parameters():
            if not torch.all(torch.isfinite(param)):
                self.log("all_isfinite", 0)
                break
        else:
            self.log("all_isfinite", 1)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.unet.parameters(), lr=0.1)
