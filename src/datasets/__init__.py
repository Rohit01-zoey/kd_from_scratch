from .mnist import MNIST_data, FashionMNIST_data
from .cifar import CIFAR10_data, CIFAR100_data
from .svhn import SVHN_data
from .stl import STL10_data
from .utils import *


__all__ = ('MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'SVHN', 'STL10')