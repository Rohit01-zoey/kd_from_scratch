import os
from torch.utils.data import Dataset
from torchvision import datasets


from datasets.cifar import CIFAR10_data, CIFAR100

obj = CIFAR10_data(root = './kd_from_scratch/src/data', split = 'train')
print(obj.data)