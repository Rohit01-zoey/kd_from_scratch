import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn
# from config import cfg


class DecConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels, kernel_size, stride, padding, bias):
        super(DecConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.sigma_weight = nn.Parameter(copy.deepcopy(self.weight.data) / 2)
        self.phi_weight = nn.Parameter(copy.deepcopy(self.weight.data) / 2)
        self.weight = None
        if bias:
            self.sigma_bias = nn.Parameter(copy.deepcopy(self.bias.data) / 2)
            self.phi_bias = nn.Parameter(copy.deepcopy(self.bias.data) / 2)
            self.bias = None
            self.bias_ = self.sigma_bias + self.phi_bias
        else:
            self.register_parameter('bias_', None)

    def forward(self, input):
        if self.bias is not None:
            return self._conv_forward(input, self.sigma_weight + self.phi_weight, self.sigma_bias + self.phi_bias)
        else:
            return self._conv_forward(input, self.sigma_weight + self.phi_weight, self.bias)