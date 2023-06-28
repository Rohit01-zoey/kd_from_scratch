import os
from torch.utils.data import Dataset
from torchvision import datasets
from data import fetch_dataset
from model import resnet18



# print(os.getcwd())
# data_name = 'MNIST'
# # obj = CIFAR10_data(root =  './src/data/', split = 'train')
# # obj = eval('{}_data(root = \'./src/data/{}\', split = \'train\')'.format(data_name, data_name))


# data = fetch_dataset(data_name)
# print(data['train'].transform)

res = resnet18()
res.summary()