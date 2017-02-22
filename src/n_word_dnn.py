#import keras 
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
nb_epoch = 10
batch_size = 1280
val_class = 'diy'
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
    file_name = data_class+'_n_word.npz.npy'
    if os.path.isfile(file_name):
        print "Exist "+file_name+" file!"
        feature = np.load(file_name).item()
    else:
        feature = util.n_word_feature(data_all[data_class])
        np.save(file_name, feature)
        print "Complete " + data_class + " feature making..."
    data[data_class] = feature
x_train = np.array([[]])
y_train = np.array([[]])
a = 0
b = 0
for data_class in data:
    if data_class == val_class:
        x_text = data[data_class]['text']
        x_id = data[data_class]['id']
        x_val = data[data_class]['x']
        y_val = data[data_class]['y']
    else:
        if x_train.shape[1] == 0:
            x_train = data[data_class]['x']
            y_train = data[data_class]['y']
        else:
            x_train = np.append(data[data_class]['x'],x_train,axis=0)
            y_train = np.append(data[data_class]['y'],y_train,axis=0)

x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train).astype('float32')
x_val = np.array(x_val).astype('float32')
y_val = np.array(y_val).astype('float32')
print "x_train: ", x_train.shape
print "y_train: ", y_train.shape
print "x_val: ", x_val.shape
print "y_val: ", y_val.shape
for yy in y_val:
    if yy[0] > yy[1]:
        a += 1
for yy in y_train:
    if yy[0] > yy[1]:
        b += 1
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
            validation_data=(x_val, y_val),
            callbacks=[checkpointer] ,
            class_weight=[1, 0.1])
# evaluate
f1_score = []
precision = []
recall = []
model.load_weights("n_word_dnn_weights.hdf5")
output_file = open('output.csv','w')
predict = model.predict(x_val)
text = ""
ans = {}
print "evaluation..."
for index, pre in enumerate(predict):
    #if pre[0] > float(a)/y_val.shape[0]:
    if pre[0] > 0.1:
        if x_id[index] not in ans:
            ans[x_id[index]]=""
        ans[x_id[index]] += x_text[index]
        ans[x_id[index]] += ' '
for index, tags in enumerate(data_all[val_class]['tags']):
    doc_id = data_all[val_class]['id'][index]
    ans_str = ""
    if doc_id in ans:
        ans_str = ans[doc_id]
    out_str = str(doc_id)+','+ans_str+'\n'
    output_file.write(out_str)
    p,r,f = evaluate.f1_score(ans_str, tags)
    f1_score.append(f)
    precision.append(p)
    recall.append(r)
    
print np.mean(precision), np.mean(recall), np.mean(f1_score)
