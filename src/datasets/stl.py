''' py file for the implementation and the loading of the stl10 dataset'''


import os
from torch.utils.data import Dataset
from torchvision import datasets


class STL10_data():
    data_name = 'STL10'
    
    def __init__(self, root, split='train', transform = None):
        """Generates an instance of the CIFAR10 dataset.

        Args:
            root (str): Root of the directory where the instance of the dataset is stored
            split (str): Split of the dataset. Can be either 'train' or 'test'
            transform (transform object, optional): Transform object of the pytorch for the transormation of the input data. Defaults to None.
        """
        self.root = os.path.expanduser(root) # expand the user path wehre the dataset is stored
        self.split = split # is set to true if the split is train else is false
        self.transform = transform # store the transform object
        self.data = datasets.STL10(root = self.root, split = self.split, transform=self.transform, download = True) # geting the cifar10 dataset from the torchvision