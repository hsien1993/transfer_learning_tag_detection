from keras.models import Model
from keras.layers import LSTM, GRU, Dense, Input, Embedding
from random import randint
from keras import optimizers
from keras.utils import np_utils
from util import clean_sent, load_data, clean_tag
from nltk import pos_tag
from evaluate import f1_score
import numpy as np
import random

def make_tag(all_tag, tags):
    temp = [0]*len(all_tag)
    for tag in tags:
        temp[ all_tag.index(tag) ] = 1
    return temp

home_dir = '/home/hsienchin/transfer_learning_tag_detection/'
data_dir = home_dir + 'data/'
feature_dir = home_dir + 'feature/'
document = load_data(data_dir, '_with_stop_words_3.csv')

# padding zero on the front
max_len = 20
val_class = 'travel'
rate = 0.5
#num = 1~1000
data = np.load('../feature/travelcnn_feature.npy').item()
whole_data = data
all_id = set(whole_data['id'])
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
all_tag = []
id2tag = {}
for index,tag in enumerate(document[val_class]['tags']):
    doc_id = document[val_class]['id'][index]
    tag = clean_tag(tag)
    id2tag[doc_id] = set(tag)
    all_tag.extend(tag)

all_tag = list(set(all_tag))


for i,t in enumerate(whole_data['id']):
    tags = id2tag[t]
    if t in val_id:
        x_val.append(whole_data['w2v'][i])
        y_val.append(make_tag(all_tag, tags))
        x_id.append(whole_data['id'][i])
    else:
        x_train.append(whole_data['w2v'][i])
        y_train.append(make_tag(all_tag, tags))

x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train).astype('float32')
x_val = np.array(x_val).astype('float32')
y_val = np.array(y_val).astype('float32')


print x_train.shape
print y_train.shape
# model
sent_in = Input(shape=(x_train[0].shape))
x = LSTM(500, return_sequences=True, activation='elu')(sent_in)
x = LSTM(500, return_sequences=False, activation='tanh')(sent_in)
x = Dense(len(all_tag), activation='softmax')(x)
#x = LSTM(2, activation='softmax', return_sequences=True)(x)
my_model = Model(input=sent_in, output=x)
my_model.summary()
RMSprop = optimizers.RMSprop(clipnorm=1, lr=0.005)
my_model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(),metrics=['accuracy'])
#my_model.compile(loss='mean_squared_error',optimizer=optimizers.Adam(),metrics=['accuracy'])
my_model.fit(x_train,y_train,
             batch_size=128,
             nb_epoch=15,
             validation_data=(x_val,y_val))

result = my_model.predict(x_val)
print result[0]

ans = {}
for index, doc_id in enumerate(x_id):
    temp = ""
    for i,value in enumerate(result[index]):
        if value > 0.005:
            temp += all_tag[i] + ' '
    if doc_id not in ans:
        ans[doc_id] = ""
    ans[doc_id] += temp
f = open('lstm.out','w')
for i in ans:
    f.write(str(i))
    f.write(',')
    f.write(ans[i])
    f.write('\n')
f.close()


precision = []
recall =[]
f1 = []
for index, tags in enumerate(document[val_class]['tags']):
    doc_id = document[val_class]['id'][index]
    pre_tag = ""
    if doc_id in ans:
        pre_tag = ans[doc_id]
    p,r,f = f1_score(pre_tag, tags)
    precision.append(p)
    recall.append(r)
    f1.append(f)
print np.mean(precision), np.mean(recall), np.mean(f1)
