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
nb_epoch = 100
batch_size = 128
val_class = 'cooking'
window_size = 2

# make data
# feature = [tf, idf, tf*idf, isTitle, wordcount, word_position]

print "*****************************"
print "**********read data**********"
print "*****************************"

data_dir = '/home/hsienchin/transfer_learning_tag_detection/data/'
title_only = True
print "Reading light data..."
data_light = util.load_data(data_dir, '_light.csv')
print "Reading whole data..."
data_with_stop_words = util.load_data(data_dir, '_with_stop_words.csv')

print "*****************************"
print "********make feature*********"
print "*****************************"

data = {}
for data_class in data_light:
    if os.path.isfile(data_class+'_n_word.npz'):
        feature_file = np.load(data_class+'_n_word.npz')
        x = feature_file['x']
        y = feature_file['y']
    else:
        x, y = util.make_feature(data_light[data_class], data_with_stop_words[data_class])
        x = np.array(x)
        y = np.array(y)
        np.savez(open(data_class+'.npz','w'),x=x,y=y)
        print "Complete " + data_class + " feature making..."
    data[data_class] = {}
    data[data_class]['x'] = x
    data[data_class]['y'] = y
    
x_train = np.array([[]])
y_train = np.array([[]])
for data_class in data:
    if data_class == val_class:
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
# model
inputs = Input( shape=(x_train.shape[1],) )
x = Dense( 52, activation='relu')(inputs)
x = BatchNormalization(axis=1)(x)
#x = Dense( 512, activation='relu')(x)
outputs = Dense( 2, activation='softmax')(x)

model = Model(input=inputs, output=outputs)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="uni_dnn_weights.hdf5", save_best_only=True, monitor='val_acc', mode='max' )
model.fit(  x_train, y_train, 
            batch_size=batch_size, 
            nb_epoch=nb_epoch, 
            validation_data=(x_val, y_val),
            callbacks=[checkpointer] ,
            class_weight=[1, 0.5])   
# evaluate
title_idf, content_idf = tf_idf.inverse_frequency(data_light[val_class], opt='smooth')
word2count = util.word_count(data_light[val_class])
f1_score = []
precision = []
recall = []
model.load_weights("uni_dnn_weights.hdf5")
output_file = open('output.csv','w')
for index, title in enumerate(data_light[val_class]['title']):
    tags = data_light[val_class]['tags'][index]
    ans = ""
    for word in title.split():
        tf = tf_idf.term_frequency(word, title)
        feature = [ tf, 
                    title_idf[word], 
                    tf*title_idf[word], 
                    1, 
                    word2count[word], 
                    util.find_position(word, data_with_stop_words[val_class]['title'][index]) ]
        prediction = model.predict(np.array([feature]).astype('float32'))
        if prediction[0,0]>prediction[0,1]:
            ans = ans + word + ' '
    output_file.write(str(data_light[val_class]['id'][index])+','+ans+'\n')
    p,r,f = evaluate.f1_score(ans,tags)
    f1_score.append(f)
    precision.append(p)
    recall.append(r)
print val_class, ' p,r,f: ', np.mean(precision), np.mean(recall), np.mean(f1_score)
output_file.close()
