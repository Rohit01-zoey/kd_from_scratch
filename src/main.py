import os
from torch.utils.data import Dataset
from torchvision import datasets
from data import fetch_dataset, make_dataloader
from model import resnet18
from config import cfg

from torch.utils.data.dataloader import default_collate



# print(os.getcwd())
data_name = cfg['data_name']
# # obj = CIFAR10_data(root =  './src/data/', split = 'train')
# # obj = eval('{}_data(root = \'./src/data/{}\', split = \'train\')'.format(data_name, data_name))


data = fetch_dataset(data_name)
# print(data['train'].transform)

res = resnet18()
# res.summary()

dataloaded = make_dataloader(data, batch_size = cfg['batch_size'], shuffle = False)
print(dataloaded['train'])

for (index, data) in enumerate(dataloaded['train']):
    input_ = {}
    input_['data'] = data[0]
    input_['target'] = data[1]
    output_ = res(input_) # getting the output of the batch