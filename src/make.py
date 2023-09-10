import argparse
import itertools
from config import cfg, process_args
from modules import *
from model import *


parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)
print(cfg)


# def main():
#     # run = args['run']
#     # data_name = args['data_name']
#     # tag = "teacher"
#     # print(cfg[tag]['hidden_size'])
#     # tacher = Teacher()
#     # print(tacher.model)
#     # #print(run, data_name)
#     process_args(args)
#     print(cfg['data_name'])
#     #odel = resnet18(tag = "teacher", momentum=None, track=False)
    
    
# if __name__ == '__main__':
#     main()