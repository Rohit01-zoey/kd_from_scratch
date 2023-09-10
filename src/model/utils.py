import torch
import torch.nn as nn
import torch.nn.functional as F
import model

from metrics import utils

def init_param(m):
    if isinstance(m, nn.Conv2d) and isinstance(m, model.DecConv2d):
        nn.init.kaiming_normal_(m.sigma_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(m.phi_weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


def make_batchnorm(m, momentum, track_running_stats):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
            m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m


def loss_fn(output, target, tag, **kwargs):
    """Defines the loss function for the forward pass depending on the tag. If the 'tag' is set to 'teacher', it returns the cross entropy loss. If the 'tag' is set to 'student', it returns the knowledge distillation loss.

    Args:
        output (tensor): The output logits returned from the forward pass
        target (tensor): The target labels of the dataset inputted to the forward pass
        tag (str): The tag of the model. Could be 'teacher' or 'student'

    Returns:
        float: The loss value depending on the tag
    """
    if tag=='teacher':
        return utils.ce_loss(output, target)
    elif tag=='student':
        return utils.kd_loss(output, target, T = kwargs['T'], alpha = kwargs['alpha'])