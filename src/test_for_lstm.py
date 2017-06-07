from keras.models import Model
from keras.layers import LSTM, GRU, Dense, Input, Embedding
from random import randint
import numpy as np
# padding zero on the front
max_len = 20
#num = 1~1000

# model
sent_in = Input(shape=(max_len,))
embed = Embedding(output_dim=512, input_dim=1000, input_length=max_len)(sent_in)
x = LSTM(32, return_sequences=True )(embed)
x = LSTM(2,return_sequences=True)(x)
my_model = Model(input=sent_in, output=x)
my_model.summary()
# make data
num_train = 5000
num_val = 1000
x_train = np.zeros((num_train,max_len))
y_train = np.zeros((num_train,max_len,2))
x_val = np.zeros((num_val,max_len))
y_val = np.zeros((num_val,max_len,2))

for i in range(0,num_train):
    temp = [0]*max_len
    y_temp = [[0, 0]]*max_len
    l = randint(1,max_len)
    for j in range(0,l):
        temp[j] = randint(1,1000-1)
        y_temp[j] = [1, 0] if temp[j] % 2 == 0 else [0,1]
    x_train[i] = temp
    y_train[i] = y_temp

for i in range(0,num_val):
    temp = [0]*max_len
    y_temp = [[0,0]]*max_len
    l = randint(1,max_len)
    for j in range(0,l):
        temp[j] = randint(1,1000-1)
        y_temp[j] = [1, 0] if temp[j] % 2 == 0 else [0,1]
    x_val[i] = temp
    y_val[i] = y_temp
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
y_train = y_train.astype('float32')
y_val = y_val.astype('float32')
print x_train.shape
print x_val.shape
my_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
my_model.fit(x_train,y_train,
             batch_size=32,
             nb_epoch=10,
             validation_data=(x_val,y_val))
print my_model.predict(np.array([[1,2,3,4,5,6,7,8,9,10,11,1,0,0,0,0,0,0,0,0]]).astype('float32'))            
