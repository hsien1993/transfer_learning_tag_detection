import util
import evaluate
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, merge
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D, MaxPooling1D, Flatten
import os

val_class = 'cooking'
word_embedding_size = 200
max_len = 20
w2v_model_name = 'my_word2vec_2.model'


print "loading data..."
data_dir = '/home/hsienchin/transfer_learning_tag_detection/data/'
data_light = util.load_data(data_dir, '_light.csv')
data_with_stop_word = util.load_data(data_dir, '_with_stop_words_2.csv')

x_train_word_embedding = np.array([[]])
x_train_tf_idf = np.array([[]])
x_train_pos_tag = np.array([[]])
y_train_hasTag = np.array([[]])
y_train_TagPosition = np.array([[]])

x_val_word_embedding = []
x_val_tf_idf = []
x_val_pos_tag = []
y_val_hasTag = []
y_val_TagPosition = []
#feature = [word_embedding] + [tf, idf, tf*idf, isTitle, isStopWord, word_count, word_position] + POS_tagging
data = {}
if os.path.isfile(data_dir+'cnn_feature_2.npy'):
    print "reading exist feature file..."
    data = np.load( data_dir+'cnn_feature.npy' ).item()
else:
    print "extracting feature..."
    for data_class in data_with_stop_word:
        x_0, x_1, x_2, y_0, y_1 = util.make_cnn_feature( data_light[data_class], 
                                                         data_with_stop_word[data_class], 
                                                         word_embedding_size=word_embedding_size, 
                                                         sent_size=max_len, 
                                                         word2vec_model_name=w2v_model_name)
        data[data_class] = {}
        data[data_class]['x_w2v'] = x_0
        data[data_class]['x_tf_idf'] = x_1
        data[data_class]['x_pos_tag'] = x_2
        data[data_class]['y_hasTag'] = y_0
        data[data_class]['y_TagPosition'] = y_1
    np.save( data_dir+'cnn_feature_2.npy', data)

for data_class in data:
    read_in = [ np.array(data[data_class]['x_w2v']), 
                np.array(data[data_class]['x_tf_idf']), 
                np.array(data[data_class]['x_pos_tag']),
                np.array(data[data_class]['y_hasTag']),
                np.array(data[data_class]['y_TagPosition']) ]
    if data_class == val_class:
        x_val_word_embedding = read_in[0]
        x_val_tf_idf = read_in[1]
        x_val_pos_tag = read_in[2]
        y_val_hasTag = read_in[3]
        y_val_TagPosition = read_in[4]

    else:
        if x_train_word_embedding.shape[1] == 0:
            x_train_word_embedding = read_in[0]
            x_train_tf_idf = read_in[1]
            x_train_pos_tag = read_in[2]
            y_train_hasTag = read_in[3]
            y_train_TagPosition = read_in[4]
        else:
            x_train_word_embedding = np.append(read_in[0],x_train_word_embedding,axis=0)
            x_train_tf_idf = np.append(read_in[1],x_train_tf_idf,axis=0)
            x_train_pos_tag = np.append(read_in[2],x_train_pos_tag,axis=0)
            y_train_hasTag = np.append(read_in[3],y_train_hasTag,axis=0)
            y_train_TagPosition = np.append(read_in[4],y_train_TagPosition,axis=0)

x_train_word_embedding = np.array(x_train_word_embedding).astype('float32')
x_train_tf_idf = np.array(x_train_tf_idf).astype('float32')
x_train_pos_tag = np.array(x_train_pos_tag).astype('float32')
y_train_hasTag = np.array(y_train_hasTag).astype('float32')
y_train_TagPosition = np.array(y_train_TagPosition).astype('float32')

# model
#tf_idf_input = Input(shape=(tf_idf_size), name='tf_idf_input')
#word_input = Input(shape=

word_embed_input = Input(shape=(max_len, word_embedding_size), name='word_embed')
tf_idf_input = Input(shape=(max_len, 7), name='tf_idf')
pos_input = Input(shape=(max_len,), name='pos')

pos_embed = Embedding(input_dim=36, output_dim=10, input_length=max_len)(pos_input)
word_embed_conv = Convolution1D(nb_filter=64, filter_length=2, border_mode='same', input_shape=(max_len, word_embedding_size))(word_embed_input)
tf_idf_conv = Convolution1D(nb_filter=64, filter_length=2, border_mode='same', input_shape=(max_len, 7))(tf_idf_input)
pos_conv = Convolution1D(nb_filter=64, filter_length=2, border_mode='same', input_shape=(max_len, 10))(pos_embed)

x = merge([word_embed_conv, tf_idf_conv, pos_conv], mode='concat')
x = Flatten()(x)
fc = Dense(output_dim=128, activation='relu')(x)
has_tag = Dense(output_dim=2, activation='softmax', name='has_tag')(fc)
tag_position = Dense(output_dim=max_len, activation='softmax', name='tag_position')(fc)

model = Model(input=[word_embed_input, tf_idf_input, pos_input], output=[has_tag, tag_position])
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 1.])
model.fit( {'word_embed':x_train_word_embedding, 'tf_idf':x_train_tf_idf, 'pos':x_train_pos_tag},
           {'has_tag':y_train_hasTag, 'tag_position':y_train_TagPosition},
           nb_epoch=50,
           batch_size=32)
         


