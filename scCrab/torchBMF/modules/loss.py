import warnings
import torch.nn as nn
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

from .. import functional as BF


def loss_amse(recons, ori):
    """
    Adapted mean square error
    Parameters
    ------
    recons
        reconstructed gene expression matrix
    ori
        original gene expression matrix
    Returns
    ------
    loss
        adapted mean square loss
    """

    SE_keepdim = nn.MSELoss(reduction="none")

    Gamma = ori.data.sign().absolute()
    Q = Gamma.mean(dim=1)
    Gamma = Gamma + (Gamma - 1).absolute() * Q.reshape(-1, 1)

    loss = SE_keepdim(ori, recons) * Gamma

    return loss.mean()



def loss_kld(mu, logvar):
    """
    KL divergence of normal distribution N(mu, exp(logvar)) and N(0, 1)
    Parameters
    ------
    mu
        mean vector of normal distribution
    logvar
        Logarithmic variance vector of normal distribution
    Returns
    ------
    KLD
        KL divergence loss
    """

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5) / mu.shape[0]

    return KLD








class _Loss(Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction
            
class BKLLoss(_Loss):
    """
    Loss for calculating KL divergence of baysian neural network model.

    Arguments:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.    
    """
    __constants__ = ['reduction']

    def __init__(self, reduction='mean', last_layer_only=False):
        super(BKLLoss, self).__init__(reduction)
        self.last_layer_only = last_layer_only

    def forward(self, model):
        """
        Arguments:
            model (nn.Module): a model to be calculated for KL-divergence.
        """
        return BF.bayesian_kl_loss(model, reduction=self.reduction, last_layer_only=self.last_layer_only)