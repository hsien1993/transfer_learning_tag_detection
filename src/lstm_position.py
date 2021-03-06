from keras.models import Model
from keras.layers import LSTM, GRU, Dense, Input, Embedding
from random import randint
from keras import optimizers
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from util import clean_sent, load_data
from nltk import pos_tag
from evaluate import f1_score
import numpy as np
import sys

# padding zero on the fron
all_class = ['biology','cooking','travel','robotics','crypto','diy']
max_len = 20
#val_class = all_class[1]
val_class = all_class[int(sys.argv[1])]
#num = 1~1000
i = 0
for data_class in all_class:
    if data_class == val_class:
        data = np.load('../feature/'+val_class+'cnn_feature.npy').item()
        x_text = data['text']
        x_id = data['id']
        x_val = np.concatenate((data['w2v'],data['tf_idf']),axis=2)
        #x_val = data['tf_idf']
        y_v = data['y_tag_position']
    else:
        data = np.load('../feature/'+data_class+'cnn_feature.npy').item()
        if i == 0:
            x_train = np.concatenate((data['w2v'],data['tf_idf']),axis=2)
            #x_train = data['tf_idf']
            y_ = np.array(data['y_tag_position'])
            i = i + 1
        else:
            _x = np.concatenate((data['w2v'],data['tf_idf']),axis=2)
            #_x = data['tf_idf']
            _y = np.array(data['y_tag_position'])
            x_train = np.concatenate((x_train,_x),axis=0)
            y_ = np.concatenate((y_,_y),axis=0)
        print '--------------------------'
        print 'y_.shape', y_.shape
        print '--------------------------'
print 'y_.shape', y_.shape
y_train = []
for y in y_ :
    y_train.append(np_utils.to_categorical(y,2))
x_train = x_train.astype('float32')
y_train = np.array(y_train).astype('float32')
print 'y_train.shape', y_train.shape
y_val = []
for y in y_v:
    y_val.append(np_utils.to_categorical(y,2))
x_val = x_val.astype('float32')
y_val = np.array(y_val).astype('float32')


print x_train[0].shape
# model
sent_in = Input(shape=(x_train[0].shape))
x = BatchNormalization(axis=1)(sent_in)
x = LSTM(50, return_sequences=True, activation='elu')(x)
x = BatchNormalization(axis=1)(x)
#x = LSTM(50, return_sequences=True, activation='elu')(x)
x = LSTM(50, return_sequences=True, activation='tanh')(x)
x = BatchNormalization(axis=1)(x)
x = Dense(20, activation='relu')(x)
x = BatchNormalization(axis=1)(x)
x = Dense(2, activation='softmax')(x)
#x = LSTM(2, activation='softmax', return_sequences=True)(x)
my_model = Model(input=sent_in, output=x)
my_model.summary()
RMSprop = optimizers.RMSprop(clipnorm=1, lr=0.0001)
my_model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(),metrics=['accuracy'])
my_model.fit(x_train,y_train,
             batch_size=1024,
             nb_epoch=5,
             validation_data=(x_val,y_val))

result = my_model.predict(x_val)
print result[0]
home_dir = '/home/hsienchin/transfer_learning_tag_detection/'
data_dir = home_dir + 'data/'
feature_dir = home_dir + 'feature/'
document = load_data(data_dir, '_with_stop_words_3.csv')

ans = {}
choose_pos = ['NN', 'NNP', 'NNS', 'VB', 'VBD', 'VBG', 'VBP', 'VBZ']
all_th = [ 0.3, 0.2, 0.1,0.05, 0.01, 0.005, 0.001, 0.0001, 0.00001]
#all_th = [0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
result_file = open('../result/' + val_class + '_lstm_position.csv', 'w')
for th in all_th:
    for index, doc_id in enumerate(x_id):
        temp = ""
        words = clean_sent(x_text[index])
        word_pos = pos_tag(words)
        for position,word in enumerate(words):
            if result[index][position][1] > th:#and word_pos[position][1] in choose_pos:
                temp += word + ' '
        if doc_id not in ans:
            ans[doc_id] = ""
        ans[doc_id] += temp

    precision = []
    recall =[]
    f1 = []
    for index, tags in enumerate(document[val_class]['tags']):
        doc_id = document[val_class]['id'][index]
        pre_tag = ""
        if doc_id in ans:
            pre_tag = ans[doc_id]
        p,r,f = f1_score(pre_tag, tags, isBigram=False)
        precision.append(p)
        recall.append(r)
        f1.append(f)
    l= str(th)+','+str(np.mean(precision))+','+ str(np.mean(recall))+','+str( np.mean(f1))
    result_file.write(l+'\n')
    print l
result_file.close()
'''
f = open('lstm.out','w')
for i in ans:
    f.write(str(i))
    f.write(',')
    f.write(ans[i])
    f.write('\n')
f.close()
'''
