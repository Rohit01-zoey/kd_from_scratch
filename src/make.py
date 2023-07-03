import argparse
import itertools
from config import cfg
from modules import *
from model import *

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--data_name', default='CIFAR10', type=str)
parser.add_argument('--teacher_model_name', default='resnet18', type=str)
args = vars(parser.parse_args())

def main():
    # run = args['run']
    # data_name = args['data_name']
    # tag = "teacher"
    # print(cfg[tag]['hidden_size'])
    # tacher = Teacher()
    # print(tacher.model)
    # #print(run, data_name)
    odel = resnet18(tag = "teacher", momentum=None, track=False)
    
    
if __name__ == '__main__':
    main()