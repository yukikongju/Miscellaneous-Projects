"""
Metrics
- Generalized F_{\beta} score
- Dice Loss
- BCE Dice Loss

Precision = TP / (TP + FP) =>
Recall =


Notes:
- why we can use generalized f-score to compute dice loss?

"""

import torch


def f_score(pr, gt, beta=1, threshold=None, eps=1e-7, activation="sigmoid"):
    """ """
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(f"Activation Implemented for sigmoid and softmax2d.")

    pr = activation_fn(pr)
    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(pr * gt)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    return score


class DiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-7, threshold=None, activation="sigmoid"):
        super().__init__()
        self.eps = eps
        self.threshold = threshold
        self.activation = activation

    def forward(self, pr, gt):
        return 1 - f_score(
            pr, gt, beta=1, threshold=self.threshold, eps=self.eps, activation=self.activation
        )


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-7, threshold=None, activation="sigmoid"):
        super().__init__()
        self.eps = eps
        self.threshold = threshold
        self.activation = activation
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, pr, gt):
        dice = 1 - f_score(
            pr, gt, beta=1, threshold=self.threshold, eps=self.eps, activation=self.activation
        )
        bce = self.bce(pr, gt.float())
        return dice + bce
