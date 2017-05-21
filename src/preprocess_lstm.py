import argparse
import numpy as np
import random
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--supervised', help='supervised at which class')
    parser.add_argument('-u','--unsupervised', help='validation at which class')
    parser.add_argument('data_dir',help='the position of data')
    return parser.parse_args()

args = parse_args()
data_dir = args.data_dir

if len(args.supervised) > 0:
    data_class = args.supervised
    data = np.load(data_dir).item
    all_id = set(data['id'])
    l = len(all_id)
    val_id = random.sample(all_id, int(l*rate))
    x_val = []
    y_val= []
    x_text = []
    x_id = []
    x_train = []
    y_train = []
    y_v = []
    y_t = []
    for i,t in enumerate(whole_data['id']):
        if t in val_id:
            x_val.append(whole_data['w2v'][i])
            y_v.append(whole_data['y_tag_position'][i])
            x_text.append(whole_data['text'][i])
            x_id.append(whole_data['id'][i])
        else:
            x_train.append(whole_data['w2v'][i])
            y_t.append(whole_data['y_tag_position'][i])
    
    print x_text[0]
    for y in y_t:
        y_train.append(np_utils.to_categorical(y,2))
    for y in y_v:
        y_val.append(np_utils.to_categorical(y,2))
    x_train = np.array(x_train).astype('float32')
    y_train = np.array(y_train).astype('float32')
    x_val = np.array(x_val).astype('float32')
    y_val = np.array(y_val).astype('float32')
    feature = {}
    feature['x_train'] = x_train
    feature['y_train'] = y_train
    feature['x_val'] = x_val
    feature['y_val'] = y_val
    feature['x_id'] = x_id
    feature['x_text'] = x_text
    np.save('feature.npy',feature)
