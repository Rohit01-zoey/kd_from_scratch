import argparse
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--data_name', default='CIFAR10', type=str)
args = vars(parser.parse_args())

def main():
    run = args['run']
    data_name = args['data_name']
    print(run, data_name)
    
    
if __name__ == '__main__':
    main()