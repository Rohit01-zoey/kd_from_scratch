''' py file for the implementation and the loading of the mnist and the fashion mnsit dataset'''
import os
from torch.utils.data import Dataset
from torchvision import datasets

class MNIST_data():
     def __init__(self, root, split='train', transform = None):
        """Generates an instance of the CIFAR100 dataset.

        Args:
            root (str): Root of the directory where the instance of the dataset is stored
            split (str): Split of the dataset. Can be either 'train' or 'test'
            transform (transform object, optional): Transform object of the pytorch for the transormation of the input data. Defaults to None.
        """
        self.root = os.path.expanduser(root) # expand the user path wehre the dataset is stored
        self.split = split=='train' # is set to true if the split is train else is false
        self.transform = transform # store the transform object
        self.data = datasets.MNIST(root = self.root, train = self.split, transform=self.transform, download = True) # geting the cifar10 dataset from the torchvision

class FashionMNIST_data():
    def __init__(self, root, split='train', transform = None):
        """Generates an instance of the CIFAR100 dataset.

        Args:
            root (str): Root of the directory where the instance of the dataset is stored
            split (str): Split of the dataset. Can be either 'train' or 'test'
            transform (transform object, optional): Transform object of the pytorch for the transormation of the input data. Defaults to None.
        """
        self.root = os.path.expanduser(root) # expand the user path wehre the dataset is stored
        self.split = split=='train' # is set to true if the split is train else is false
        self.transform = transform # store the transform object
        self.data = datasets.FashionMNIST(root = self.root, train = self.split, transform=self.transform, download = True) # geting the cifar10 dataset from the torchvision