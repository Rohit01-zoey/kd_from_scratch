''' py file for the implementation and the loading of the cifar10 as well as the cifar100 dataset'''

import os
from torch.utils.data import Dataset
from torchvision import datasets

class CIFAR10_data():
    data_name = 'CIFAR10'
    class_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    
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
        self.data = datasets.CIFAR10(root = self.root, train = self.split, transform=self.transform, download = True) # geting the cifar10 dataset from the torchvision



class CIFAR100_data():
    data_name = 'CIFAR100'
    class_name = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    }
    
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
        self.data = datasets.CIFAR100(root = self.root, train = self.split, transform=self.transform, download = True) # geting the cifar10 dataset from the torchvision

