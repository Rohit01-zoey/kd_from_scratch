''' py file for the implementation and the loading of the cifar10 as well as the cifar100 dataset'''

import os
from torch.utils.data import Dataset
from torchvision import datasets

class CIFAR10_data():
    data_name = 'CIFAR10'
    
    
    def __init__(self, root, split='train', transform = None):
        """Generates an instance of the CIFAR10 dataset.

        Args:
            root (str): Root of the directory where the instance of the dataset is stored
            split (str): Split of the dataset. Can be either 'train' or 'test'
            transform (transform object, optional): Transform object of the pytorch for the transormation of the input data. Defaults to None.
        """
        self.root = os.path.expanduser(root) # expand the user path wehre the dataset is stored
        self.split = split=='train' # is set to true if the split is train else is false
        self.transform = transform # store the transform object
        self.data = datasets.CIFAR10(root = self.root, train = self.split, transform=self.transform) # geting the cifar10 dataset from the torchvision



class CIFAR100(Dataset):
    data_name = 'CIFAR100'
    pass

