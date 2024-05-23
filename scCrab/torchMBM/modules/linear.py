import math
import torch
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn


class MaskedScaleDotProductModule(Module):
    """
    Module to construct MaskedScaleDotProductAttention(self-attention)
    """

    def __init__(self, scale_factor, gamma=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.gamma = gamma

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.scale_factor, k.transpose(-2, -1))

        
        attn = F.softmax(attn, dim=-1) * self.gamma
        output = torch.matmul(attn, v) + v

        return output, attn


class MaskedScaleDotProductAttention(Module):
    """
    Self-attention
    """

    def __init__(self, d_in, d_out, d_k, gamma=1.0, dropout=0.1):
        super().__init__()

        self.d_k = d_k

        self.w_qs = nn.Linear(d_in, d_k, bias=False)
        self.w_ks = nn.Linear(d_in, d_k, bias=False)
        self.fc = nn.Linear(d_in, d_out, bias=False)

        self.attention = MaskedScaleDotProductModule(scale_factor=d_k ** 0.5, gamma=gamma)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)

    def forward(self, q, k, v):
        q = self.w_qs(q)
        k = self.w_ks(k)

        v, attn = self.attention(q, k, v)

        v = self.dropout(self.fc(v))

        v = self.layer_norm(v)

        return v, attn





class BayesLinear(Module):
    r"""
    Applies Bayesian Linear
    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.
    .. note:: other arguments are following linear of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    
    """
    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)
        
        self.bias = bias
        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        

    def reset_parameters(self, prior_mu, prior_log_sigma):
        self.weight_mu.data = prior_mu
        print('resetting params')
        self.weight_log_sigma.data = torch.ones_like(self.weight_log_sigma.data) * prior_log_sigma.to(
            self.weight_log_sigma.data.device)
        self.prior_mu = prior_mu
        self.prior_log_sigma = prior_log_sigma
        

    def freeze(self) :
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias :
            self.bias_eps = torch.randn_like(self.bias_log_sigma)
        
        
    def unfreeze(self) :
        self.weight_eps = None
        if self.bias :
            self.bias_eps = None 
            
            
    def _kld_loss(self, mu_0, log_sigma_0, mu_1, log_sigma_1):
        kl = log_sigma_1 - log_sigma_0 + \
             (torch.exp(log_sigma_0) ** 2 + (mu_0 - mu_1) ** 2) / (2 * torch.exp(log_sigma_1) ** 2) - 0.5
        return kl.mean()
    

    def bayesian_kld_loss(self):
        device = self.weight_mu.data.device
        return self._kld_loss(self.weight_mu, self.weight_log_sigma, self.prior_mu.to(device),
                              self.prior_log_sigma.to(device))
    
    
    def forward(self, x):
        device = self.weight_mu.data.device
        if self.weight_eps is None:
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma).to(
                device)
        else:
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps.to(device)

        bias = None

        return F.linear(x, weight, bias)
    
    
    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu, self.prior_sigma, self.in_features, self.out_features, self.bias is not None)