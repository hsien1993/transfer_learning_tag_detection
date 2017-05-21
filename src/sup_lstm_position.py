from keras.models import Model
from keras.layers import LSTM, GRU, Dense, Input, Embedding
from random import randint
from keras import optimizers
from keras.utils import np_utils
from util import clean_sent, load_data
from nltk import pos_tag
from evaluate import f1_score
import numpy as np
import random
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


print x_train[0].shape
# model
sent_in = Input(shape=(x_train[0].shape))
x = LSTM(500, return_sequences=True)(sent_in)
#x = Dense(2, activation='softmax')(x)
x = LSTM(2, activation='softmax', return_sequences=True)(x)
my_model = Model(input=sent_in, output=x)
my_model.summary()
RMSprop = optimizers.RMSprop(clipnorm=1, lr=0.00005)
my_model.compile(loss='categorical_crossentropy',optimizer=RMSprop,metrics=['accuracy'])
my_model.fit(x_train,y_train,
             batch_size=128,
             nb_epoch=5,
             validation_data=(x_val,y_val))

result = my_model.predict(x_val)
print result[0]
ans = {}
choose_pos = ['NN', 'NNP', 'NNS', 'VB', 'VBD', 'VBG', 'VBP', 'VBZ']
for index, doc_id in enumerate(x_id):
    temp = ""
    words = clean_sent(x_text[index])
    word_pos = pos_tag(words)
    for position,word in enumerate(words):
        if result[index][position][1] > 0.005 and word_pos[position][1] in choose_pos:
            temp += word + ' '
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

home_dir = '/home/hsienchin/transfer_learning_tag_detection/'
data_dir = home_dir + 'data/'
feature_dir = home_dir + 'feature/'

document = load_data(data_dir, '_with_stop_words_3.csv')

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
