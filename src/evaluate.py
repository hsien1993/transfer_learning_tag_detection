import pandas as pd
import re
import argparse
def f1_score(ans, ref, isBigram=False):
    prediction = ans.split()
    reference = []
    if isBigram:
        reference = ref.split()
    else:
        reference = re.split(' |-',ref)
    prediction = set(prediction)
    reference = set(reference)
    tp = len(prediction & reference)
    fp = len(prediction) - tp
    fn = len(reference) - tp
    if len(prediction) == 0:
        return 0,0,0
    if tp == 0:
        return 0,0,0
    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    return precision, recall, 2*((precision*recall)/(precision+recall))    
'''
data_dir = '/home/hsienchin/transfer_learning_tag_detection/data/'
with_out_stopwords = '_light.csv'
dataframes = {
    "cooking": pd.read_csv(data_dir + "cooking" + with_out_stopwords),
    "crypto": pd.read_csv(data_dir + "crypto" + with_out_stopwords),
    "robotics": pd.read_csv(data_dir + "robotics" + with_out_stopwords),
    "biology": pd.read_csv(data_dir + "biology" + with_out_stopwords),
    "travel": pd.read_csv(data_dir + "travel" + with_out_stopwords),
    "diy": pd.read_csv(data_dir + "diy" + with_out_stopwords),
}
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ans_file', help='ans file, include id,prediction,reference')
    return parser.parse_args()

if __name__ == '__main__':
    import numpy
    args = parse_args()
    p = []
    r = []
    f1 = []
    with open(args.ans_file) as f:
        for line in f:
            line = line.strip()
            index, prediction, reference = line.split(',')
            _p, _r, _f1 = f1_score(prediction, reference)
            p.append(_p)
            r.append(_r)
            f1.append(_f1)
    print numpy.mean(p), numpy.mean(r), numpy.mean(f1)

