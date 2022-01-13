import torch
from torch import nn
from torch.nn import functional as F


def cross_entropy(prd, tgt, weights=None):

    if prd.dim() > 2:
        N, C, H, W = prd.shape
        prd = prd.view(N, C, -1)
        prd = prd.transpose(1, 2)
        prd = prd.contiguous().view(-1, C)

    tgt = tgt.view(-1, 1)

    logp = F.log_softmax(prd)
    logp = logp.gather(1, tgt)
    logp = logp.view(-1)

    if weights is not None:
        weights = weights.gather(0, tgt.view(-1))
        logp = weights * logp
        loss = logp.sum() / weights.sum()
    else:
        loss = logp.mean()

    return -loss


def focal_loss(prd, tgt, gamma=2, weights=None):

    if prd.dim() > 2:
        N, C, H, W = prd.shape
        prd = prd.view(N, C, -1)
        prd = prd.transpose(1, 2)
        prd = prd.contiguous().view(-1, C)

    tgt = tgt.view(-1, 1)

    logp = F.log_softmax(prd)
    logp = logp.gather(1, tgt)
    logp = logp.view(-1)
    p = logp.exp()

    loss = (1 - p) ** gamma * logp

    if weights is not None:
        weights = weights.gather(0, tgt.view(-1))
        loss = weights * loss
        loss = loss.sum() / weights.sum()
    else:
        loss = loss.mean()

    return -loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weights=None):
        super().__init__()
        self.gamma = gamma
        self.weights = weights

    def forward(self, prd, tgt):
        if prd.dim() > 2:
            N, C, H, W = prd.shape
            prd = prd.view(N, C, -1)
            prd = prd.transpose(1, 2)
            prd = prd.contiguous().view(-1, C)
        tgt = tgt.view(-1, 1)

        logp = F.log_softmax(prd, dim=1)
        logp = logp.gather(1, tgt)
        logp = logp.view(-1)
        p = logp.exp()

        loss = (1 - p) ** self.gamma * logp

        if self.weights is not None:
            weights = self.weights.gather(0, tgt.view(-1))
            loss = weights * loss
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.mean()
        return -loss


def dice(y, prd_logits, eps=1):
    bs, n_cls, w, h = prd_logits.shape
    p = F.softmax(prd_logits)
    mask = torch.zeros_like(prd_logits)
    mask = mask.scatter_(1, y.view(bs, 1, w, h), 1)
    dice_loss = (2 * (mask * p).sum() + eps) / (mask.sum() + p.sum() + eps)
    return dice_loss


def L_Dice(y, prd_logits, eps=1, gamma=1, reduce="mean"):
    bs, n_cls, w, h = prd_logits.shape
    p = F.softmax(prd_logits)
    mask = torch.zeros_like(prd_logits)
    mask = mask.scatter_(1, y.view(bs, 1, w, h), 1)
    dice_loss = (2 * (mask * p).sum(dim=(2, 3)) + eps) / (
        mask.sum(dim=(2, 3)) + p.sum(dim=(2, 3)) + eps
    )
    dice_loss = (dice_loss ** gamma).mean(dim=1)  # mean over classes
    if reduce == "mean":
        dice_loss = dice_loss.mean()
    return dice_loss


def L_Cross(y, prd, weights=None, gamma=0.3, reduce="mean"):
    bs, n_cls, w, h = prd.shape
    logp = F.log_softmax(prd, dim=1)
    mask = torch.zeros_like(logp)
    mask = mask.scatter_(1, y.view(bs, 1, w, h), 1)
    loss = (mask * logp).sum(dim=(2, 3)) / mask.sum(dim=(2, 3))
    loss = (-loss) ** gamma

    if weights is not None:
        loss = (weights * loss).sum(dim=1)
    else:
        loss = loss.mean(dim=1)

    if reduce == "mean":
        loss = loss.mean()
    return loss


class ExponentialLogarithmicLoss(nn.Module):
    def __init__(
        self,
        gamma_dice=1,
        gamma_cross=1,
        w_dice=0.5,
        w_cross=0.5,
        weights_cross=None,
        eps=1,
    ):
        super().__init__()
        assert w_dice + w_cross == 1
        self.gamma_dice = gamma_dice
        self.gamma_cross = gamma_cross
        self.w_dice = w_dice
        self.w_cross = w_cross
        self.weights_cross = weights_cross
        self.eps = eps

    def cross_entropy(self, prd, y_gt):
        if prd.dim() > 2:
            N, C, H, W = prd.shape
            prd = prd.view(N, C, -1)
            prd = prd.transpose(1, 2)
            prd = prd.contiguous().view(-1, C)
        y_gt = y_gt.view(-1, 1)

        logp = F.log_softmax(prd, dim=1)
        logp = logp.gather(1, y_gt)
        logp = logp.view(-1)

        loss = (-logp) ** self.gamma_cross

        if self.weights_cross is not None:
            weights = self.weights_cross.gather(0, y_gt.view(-1))
            loss = weights * loss
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.mean()
        return loss


class MixedLoss(nn.Module):
    def __init__(
        self, gamma_dice=1, gamma_cross=1, alpha=0.5, weights_cross=None, eps=1
    ):
        super().__init__()
        self.gamma_dice = gamma_dice
        self.gamma_cross = gamma_cross
        self.alpha = alpha
        self.weights_cross = weights_cross
        self.eps = eps

    def cross_entropy(self, prd, y_gt):
        if prd.dim() > 2:
            N, C, H, W = prd.shape
            prd = prd.view(N, C, -1)
            prd = prd.transpose(1, 2)
            prd = prd.contiguous().view(-1, C)
        y_gt = y_gt.view(-1, 1)

        logp = F.log_softmax(prd, dim=1)
        logp = logp.gather(1, y_gt)
        logp = logp.view(-1)

        loss = (-logp) ** self.gamma_cross

        if self.weights_cross is not None:
            weights = self.weights_cross.gather(0, y_gt.view(-1))
            loss = weights * loss
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.mean()
        return loss

    def dice(self, prd, y_gt):
        """prd - logits"""
        _, _, W, H = prd.shape
        p = F.softmax(prd, dim=1)
        mask = torch.zeros_like(prd)
        mask = mask.scatter_(1, y_gt.view(-1, 1, W, H), 1)
        dice = 1 - (2 * (mask * p).sum(dim=(2, 3)) + self.eps) / (
            mask.sum(dim=(2, 3)) + p.sum(dim=(2, 3)) + self.eps
        )
        return dice.mean()

    def forward(self, prd, y_gt):
        loss = self.alpha * self.cross_entropy(prd, y_gt) + (
            1 - self.alpha
        ) * self.dice(prd, y_gt)
        return loss
