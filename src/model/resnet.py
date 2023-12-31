import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn
from config import cfg


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(Block, self).__init__()
        self.n1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride):
        super(Bottleneck, self).__init__()
        self.n1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.n2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(out)))
        out = self.conv3(F.relu(self.n3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, target_size):
        super().__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2)
        self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion)
        self.linear = nn.Linear(hidden_size[3] * block.expansion, target_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def f(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(self.n4(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def forward(self, input, tag, **kwargs):
        """Performs the forward pass of the model and returns the output as dictionary with the logits as well as the loss if the tag is 'teacher' or 'student'

        Args:
            input (dict): Dictionary containing the input data and the target
            tag (str): The tag of the model. Should be 'teacher' or 'student'

        Raises:
            ValueError: Raise error if the tag is not 'teacher' or 'student'

        Returns:
            dict: Returns the output as dictionary with the logits (key = 'target') as well as the loss (key = 'loss')
        """
        output = {}
        output['target'] = F.softmax(self.f(input['data']), dim = -1) # take the softmax to get the logits
        if tag in ['teacher', 'student']:
            output['loss'] = loss_fn(output['target'], input['target'], **kwargs)
        else:
            raise ValueError("Not a valid tag. The tag should be 'teacher' or 'student'")
        return output


def resnet9(tag = "teacher", momentum=None, track=False):
    """Implements the ResNet9 architecture.

    Args:
        momentum (_type_, optional): Momentum for making batch norm. Defaults to None.
        track (bool, optional): True to track the running stats. Defaults to False.

    Returns:
        model : model of the required ResNet architecture
    """
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg[tag]['hidden_size']
    model = ResNet(data_shape, hidden_size, Block, [1, 1, 1, 1], target_size)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model


def resnet18(tag = "teacher", momentum=None, track=False):
    """Implements the ResNet18 architecture.

    Args:
        tag(str): tag for the model.Required, could be 'teacher'|'student'
        momentum (_type_, optional): Momentum for making batch norm. Defaults to None.
        track (bool, optional): True to track the running stats. Defaults to False.

    Returns:
        model : model of the required ResNet architecture
    """
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg[tag]['hidden_size']
    model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], target_size)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model