import argparse
from util import load_data
from random import random
parser = argparse.ArgumentParser(description='split data to train and test')
parser.add_argument('--indir', default='../data/', help='input data dir')
parser.add_argument('--outdir', default='../data/', help='output data dir')
parser.add_argument('--type', default='.csv', help='data type')
parser.add_argument('-r','--rate', default=0.5, type=float, help='rate of train/all')
args = parser.parse_args()

#data_topics = ['biology','cooking','travel','robotics','crypto','diy']
in_dir = args.indir
out_dir = args.outdir
data_type = args.type
rate = args.rate

'''
for topic in data_topics:
    input_file = args.indir + topic + args.type
    train_file = args.outdir + 'train_' + topic + args.type
    test_file = args.outdir + 'test_' + topic + args.type
    train = open(train_file, 'w')
    test = open(test_file, 'w')
    for line in open(input_file):
        if random() < args.rate:
            train.write(line)
        else:
            test.write(line)
    train.close()
    test.close()
            
'''    
