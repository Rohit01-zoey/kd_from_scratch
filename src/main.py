import os
from torch.utils.data import Dataset
from torchvision import datasets
from data import fetch_dataset




print(os.getcwd())
data_name = 'STL10'
# obj = CIFAR10_data(root =  './src/data/', split = 'train')
# obj = eval('{}_data(root = \'./src/data/{}\', split = \'train\')'.format(data_name, data_name))


data = fetch_dataset(data_name)
print(data['train'].transform)