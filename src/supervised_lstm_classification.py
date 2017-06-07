from keras.models import Model
from keras.layers import LSTM, GRU, Dense, Input, Embedding, Dropout
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from random import randint
from keras import optimizers
from keras.utils import np_utils
from util import clean_sent, load_data, clean_tag
from nltk import pos_tag
from evaluate import f1_score
import numpy as np
import random
from collections import Counter
import operator
import sys
nb_epoch = 10
def make_tag(all_tag, input_tags):
    one_hot = []
    for tags in input_tags:
        temp = [0]*len(all_tag)
        for tag in tags.split():
            if tag in all_tag:
                temp[ all_tag.index(tag) ] = 1
        one_hot.append(temp)
    return one_hot

home_dir = '/home/hsienchin/transfer_learning_tag_detection/'
data_dir = home_dir + 'data/'
feature_dir = home_dir + 'feature/'
document = load_data(data_dir, '_light.csv')
all_class = ['biology','cooking','travel','robotics','crypto','diy']
# padding zero on the front
max_len = 20
val_class = all_class[ int(sys.argv[1]) ]
rate = 0.1

tokenizer = Tokenizer()
all_sent = [str(sent) for sent in document[val_class]['content'] ]
tokenizer.fit_on_texts(all_sent)
x = tokenizer.texts_to_sequences(all_sent)
x = pad_sequences(np.array(x),maxlen=max_len)
print x[0]
#num = 1~1000
all_id = set(document[val_class]['id'])
l = len(all_id)
val_id = random.sample(all_id, int(l*rate))
x_val = []
y_val= []
x_text = []
x_id = []
x_train = []
y_train = []
all_tag = []
tag_counter = {}
id2tag = {}
for index,tag in enumerate(document[val_class]['tags']):
    doc_id = document[val_class]['id'][index]
    tag = tag.split()
    id2tag[doc_id] = set(tag)
    all_tag.extend(tag)
all_tag = Counter(all_tag)
all_tag = [ a[0] for a in sorted(all_tag.items(), key=operator.itemgetter(1), reverse=True) if a[1] > 5 ]

#all_tag = list(set(all_tag))
all_tag_size = len(all_tag)
tags = make_tag(all_tag, document[val_class]['tags'])
'''
t = Tokenizer()
t.fit_on_texts(document[val_class]['tags'])
all_tag_size = int(len(t.word_index)*0.5)
tag_tokenizer = Tokenizer(nb_words=all_tag_size)
tag_tokenizer.fit_on_texts(document[val_class]['tags'])
tags = tag_tokenizer.texts_to_matrix(document[val_class]['tags'])
'''

for i,t in enumerate(document[val_class]['id']):
    if t in val_id:
        x_val.append(x[i])
        y_val.append(tags[i])
        x_id.append(t)
        x_text.append(text_to_word_sequence(document[val_class]['content'][i]))
    else:
        x_train.append(x[i])
        #y_train.append(make_tag(all_tag_size, tags[i]))
        y_train.append(tags[i])

x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train).astype('float32')
x_val = np.array(x_val).astype('float32')
y_val = np.array(y_val).astype('float32')


print x_train.shape
#print y_train.shape
# model
sent_in = Input(shape=(max_len,))
print 'word_size', len(tokenizer.word_counts)
embedding = Embedding(input_dim=len(tokenizer.word_counts)+1, 
                    output_dim=100, 
                    mask_zero=True, 
                    input_length=max_len)(sent_in)
x = LSTM(200, activation='elu')(embedding)
x = Dropout(0.2)(x)
#x = LSTM(100, return_sequences=False, activation='tanh')(x)
#x = LSTM(2, activation='softmax', return_sequences=True)(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.2)(x)
output_layer = Dense(all_tag_size, activation='softmax')(x)
#output_layer = [Dense(2,activation='softmax')(x)]*len(all_tag)
#output_layer = [Dense(2,activation='softmax')(x) for a in range(len(all_tag)) ] 

my_model = Model(input=sent_in, output=output_layer)
my_model.summary()
#RMSprop = optimizers.RMSprop(clipnorm=1, lr=0.005)
my_model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(),metrics=['accuracy'])
#my_model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(),metrics=['accuracy'])
my_model.fit(x_train,y_train,
             batch_size=128,
             nb_epoch=nb_epoch,
             validation_data=(x_val,y_val))

result = my_model.predict(x_val)
print result[0]

ans = {}
#tag_dict = [x[0] for x in sorted(tag_tokenizer.word_index.items(), key=operator.itemgetter(1))]
#print tag_dict[0]
g = open('../result/'+val_class+'_supervised_lstm.csv','w')
all_th = [ 1, 2, 3, 4, 5, 6, 7, 8]
for th in all_th:
    ans = {}
    for index, doc_id in enumerate(x_id):
        temp = ""
        t = result[index]
        sort_id = sorted(range(len(t)), key=lambda k:t[k], reverse=True)
        for tag_id in sort_id[:th+1]:
            temp += all_tag[tag_id] + ' '
        if doc_id not in ans:
            ans[doc_id] = ""
        ans[doc_id] += temp
        #print doc_id, ans[doc_id]

    precision = []
    recall =[]
    f1 = []
    #ans_file = open('lstm_'+str(th)+'.out','w')

    for index, tags in enumerate(document[val_class]['tags']):
        doc_id = document[val_class]['id'][index]
        pre_tag = ""
        tags = ' '.join([t for t in tags.split() if t in all_tag])
        if doc_id in ans:
            pre_tag = ans[doc_id]
            #ans_file.write(str(index)+', '+pre_tag+', '+tags+'\n')
            p,r,f = f1_score(pre_tag, tags, isBigram=True)
            precision.append(p)
            recall.append(r)
            f1.append(f)
    print np.mean(f1)
    g.write(str(th)+','+str(np.mean(precision))+','+str(np.mean(recall))+','+str(np.mean(f1))+'\n')
    #ans_file.close()
g.close()
'''
for i in ans:
    f.write(str(i))
    f.write(',')
    f.write(ans[i])
    f.write('\n')
f.close()
'''
