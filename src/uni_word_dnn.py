#import keras 
import numpy as np
import evaluate
import tf_idf
from keras.models import Model
from keras.layers import Dense, Input, Activation
from keras.callbacks import ModelCheckpoint
import pandas as pd
import random
import util
nb_epoch = 100
batch_size = 128
val_class = 'cooking'

# make data
# feature = [tf, idf, tf*idf, isTitle, wordcount, word_position]

print "*****************************"
print "**********read data**********"
print "*****************************"

data_dir = '/home/hsienchin/transfer_learning_tag_detection/data/'
title_only = True
data_light = util.load_data(data_dir, '_light.csv')
data_with_stop_words = util.load_data(data_dir, 'with_stop_words.csv')

print "*****************************"
print "********make feature*********"
print "*****************************"

x_train = []
y_train = []
x_val = []
y_val = []
x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train).astype('float32')
x_val = np.array(x_val).astype('float32')
y_val = np.array(y_val).astype('float32')
print "x_train: ", x_train.shape
print "y_train: ", y_train.shape
# model
inputs = Input( shape=(x_train.shape[1],) )
x = Dense( 128, activation='relu')(inputs)
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
word2count = word_count(data_light[val_class])
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
                    find_position(word, data_with_stop_words[val_class]['title'][index]) ]
        prediction = model.predict(np.array([feature]).astype('float32'))
        if prediction[0,0]/prediction[0,1] > 4:
            ans = ans + word + ' '
    output_file.write(str(data_light[val_class]['id'][index])+','+ans+'\n')
    p,r,f = evaluate.f1_score(ans,tags)
    f1_score.append(f)
    precision.append(p)
    recall.append(r)
print 'p,r,f: ', np.mean(precision), np.mean(recall), np.mean(f1_score)
output_file.close()
