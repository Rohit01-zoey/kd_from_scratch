''' py file for the implementation and the loading of the datasets along with the data augmentation'''


import copy
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate




data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
              'STL10': ((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687))}


def fetch_dataset(data_name):
    import datasets
    dataset = {}
    print('Fetching data {}...'.format(data_name))
    root = './src/data/{}'.format(data_name)
    if data_name in ['MNIST', 'FashionMNIST']:
        transform_train = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        transform_test = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['train'] = eval('datasets.{}_data(root=root, split=\'train\', '
                                'transform=transform_train)'.format(data_name))
        dataset['test'] = eval('datasets.{}_data(root=root, split=\'test\', '
                               'transform=transform_test)'.format(data_name))
    elif data_name in ['CIFAR10', 'CIFAR100']:
        transform_train = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        transform_test = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['train'] = eval('datasets.{}_data(root=root, split=\'train\', '
                                'transform=transform_train)'.format(data_name))
        dataset['test'] = eval('datasets.{}_data(root=root, split=\'test\', '
                               'transform=transform_test)'.format(data_name))
    elif data_name in ['SVHN']:
        transform_train = datasets.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        transform_test = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['train'] = eval('datasets.{}_data(root=root, split=\'train\', '
                                'transform=transform_train)'.format(data_name))
        dataset['test'] = eval('datasets.{}_data(root=root, split=\'test\', '
                               'transform=transform_test)'.format(data_name))
    elif data_name in ['STL10']:
        transform_train = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        transform_test = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['train'] = eval('datasets.{}_data(root=root, split=\'train\', '
                                'transform=transform_train)'.format(data_name))
        dataset['test'] = eval('datasets.{}_data(root=root, split=\'test\', '
                               'transform=transform_test)'.format(data_name))
    else:
        raise ValueError('Not valid dataset name')
    print('Data ready')
    return dataset