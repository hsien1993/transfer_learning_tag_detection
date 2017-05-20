#import keras 
#import theano
#theano.config.device = 'gpu1'
#theano.config.floatX = 'float32'
import numpy as np
import evaluate
import tf_idf
from keras.models import Model
from keras.layers import Dense, Input, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import pandas as pd
import random
import util
import os
from nltk import word_tokenize
rate = 0.5
nb_epoch = 25
batch_size = 128
val_class = 'travel'
thr = 0.12
window_size = 2

# make data
# feature = [tf, idf, tf*idf, isTitle, wordcount, word_position]

print "*****************************"
print "**********read data**********"
print "*****************************"

data_dir = '/home/hsienchin/transfer_learning_tag_detection/data/'
title_only = True
print "Reading data..."
data_all = util.load_data(data_dir, '_with_stop_words_3.csv')

print "*****************************"
print "********make feature*********"
print "*****************************"

data = {}
for data_class in data_all:
    if data_class == val_class:
        file_name = '../feature/'+data_class+'_n_word.npz.npy'
        if os.path.isfile(file_name):
            print "Exist "+file_name+" file!"
            feature = np.load(file_name).item()
        else:
            feature = util.n_word_feature(data_all[data_class])
            np.save(file_name, feature)
            print "Complete " + data_class + " feature making..."
        data[data_class] = feature
a = 0
b = 0

whole_data = data[val_class]
all_id = set(whole_data['id'])
l=len(all_id)
print l*rate
val_id = random.sample(all_id, int(l*rate))
x_val = []
y_val = []
x_text = []
x_id = []
x_train = []
y_train = []
for i,t in enumerate(whole_data['id']):
    if t in val_id:
        x_val.append(whole_data['x'][i])
        y_val.append(whole_data['y'][i])
        x_text.append(whole_data['text'][i])
        x_id.append(whole_data['id'][i])
    else:
        x_train.append(whole_data['x'][i])
        y_train.append(whole_data['y'][i])

x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train).astype('float32')
x_val = np.array(x_val).astype('float32')
y_val = np.array(y_val).astype('float32')
print "x_train: ", x_train.shape
print "y_train: ", y_train.shape
print "x_val: ", x_val.shape
print "y_val: ", y_val.shape
sample_weight = []
val_sample_weight = []
for yy in y_val:
    if yy[0] > yy[1]:
        a += 1
        val_sample_weight.append(1)
    else:
        val_sample_weight.append(0.9)
for yy in y_train:
    if yy[0] > yy[1]:
        b += 1
        sample_weight.append(1)
    else:
        sample_weight.append(0.9)
val_sample_weight = np.array(val_sample_weight)
sample_weight = np.array(sample_weight)
print a, y_val.shape[0] - a
print b, y_train.shape[0] - b
# model
inputs = Input( shape=(x_train.shape[1],) )
x = BatchNormalization(axis=1)(inputs)
#x = Dense( 50, activation='tanh')(inputs)
x = Dense( 50, activation='tanh')(x)
outputs = Dense( 2, activation='softmax')(x)

model = Model(input=inputs, output=outputs)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="n_word_dnn_weights.hdf5", save_best_only=True, monitor='val_acc', mode='max' )
model.fit(  x_train, y_train, 
            batch_size=batch_size, 
            nb_epoch=nb_epoch, 
            sample_weight=sample_weight,
            validation_data=(x_val, y_val, val_sample_weight),
            callbacks=[checkpointer] ,
            class_weight=[1, 1])
# evaluate
f1_score = []
precision = []
recall = []
model.load_weights("n_word_dnn_weights.hdf5")
output_file = open('a_n_word_output.csv','w')
predict = model.predict(x_val)
text = ""
ans = {}
print "evaluation..."
for index, pre in enumerate(predict):
    #if pre[0] > float(a)/y_val.shape[0]:
    if pre[0] > thr:
        if x_id[index] not in ans:
            ans[x_id[index]]=""
        if x_text[index] not in util.stopwords_set:
            if not x_text[index].isdigit():
                ans[x_id[index]] += x_text[index] + ' '
for index, tags in enumerate(data_all[val_class]['tags']):
    doc_id = data_all[val_class]['id'][index]
    ans_str = ""
    if doc_id in ans:
        ans_str = ' '.join(set(word_tokenize(ans[doc_id])))
    out_str = str(doc_id)+','+ans_str+'\n'
    output_file.write(out_str)
    p,r,f = evaluate.f1_score(ans_str, tags)
    f1_score.append(f)
    precision.append(p)
    recall.append(r)
    
print np.mean(precision), np.mean(recall), np.mean(f1_score)
