import util
import evaluate
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input, Reshape, merge
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.layers import Convolution2D
import os
import random
from nltk import word_tokenize
rate = 0.5
choose_th = 0.005
acc_th = 0.1

val_class = 'travel'
word_embedding_size = 200
max_len = 20
w2v_model_name = 'my_word2vec_2.model'
feature_size = 7

########### model ##############
word_embed_input = Input(shape=(1,max_len, word_embedding_size), name='word_embed')
tf_idf_input = Input(shape=(1, max_len, 7), name='tf_idf')
pos_input = Input(shape=(max_len,), name='pos')

pos_embed = Embedding(input_dim=37, output_dim=10, input_length=max_len)(pos_input)
pos_embed = Reshape((1, max_len, 10))(pos_embed)
word_embed_conv = Convolution2D(nb_filter=64, nb_row=1, nb_col=200, border_mode='valid', )(word_embed_input)
tf_idf_conv = Convolution2D(nb_filter=64, nb_row=1, nb_col=feature_size, border_mode='valid', )(tf_idf_input)
pos_conv = Convolution2D(nb_filter=64, nb_row=1, nb_col=10, border_mode='valid')(pos_embed)

x = merge([word_embed_conv, tf_idf_conv, pos_conv],mode='sum')
#x = Add([word_embed_conv, tf_idf_conv, pos_conv])
#x = merge([word_embed_conv, tf_idf_conv], mode='concat')
x = Flatten()(x)
fc = Dense(output_dim=64, activation='relu')(x)
fc = Dense(output_dim=20, activation='relu')(fc)
has_tag = Dense(output_dim=2, activation='softmax', name='has_tag')(fc)
tag_position = Dense(output_dim=max_len, activation='softmax', name='tag_position')(fc)

model = Model(input=[word_embed_input, tf_idf_input, pos_input], output=[has_tag, tag_position])
#model = Model(input=[word_embed_input, tf_idf_input], output=[has_tag, tag_position])
#model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'],
              loss_weights=[1., 1.])

print "loading data..."
home_dir = '/home/hsienchin/transfer_learning_tag_detection/'
data_dir = home_dir + 'data/'
feature_dir = home_dir + 'feature/'

document = util.load_data(data_dir, '_with_stop_words_3.csv')

x_train_w2v = []
x_train_tf_idf = []
x_train_pos = []
y_train_has_tag = []
y_train_tag_position = []

x_val_w2v = []
x_val_tf_idf = []
x_val_pos = []
x_val_id = []
x_val_text = []
y_val_has_tag = []
y_val_tag_position = []

#feature = [word_embedding] + [tf, idf, tf*idf, isTitle, isStopWord, word_count, word_position] + POS_tagging
data = {}
data_class = val_class
if os.path.isfile(feature_dir+data_class+'cnn_feature.npy'):
    print "reading exist feature file..."
    data[data_class] = np.load( feature_dir+data_class+'cnn_feature.npy' ).item()
else:
    print "extracting "+data_class+" feature..."
    data[data_class] = util.make_cnn_feature( document[data_class], 
                                              word_embedding_size=word_embedding_size, 
                                              sent_size=max_len, 
                                              word2vec_model_name=w2v_model_name)
    np.save( feature_dir+data_class+'cnn_feature.npy', data[data_class])

whole_data = data[val_class]
all_id = set(whole_data['id'])
l = len(all_id)
val_id = random.sample(all_id, int(l*rate))

for i,t in enumerate(whole_data['id']):
    if t in val_id:
        x_val_id.append(whole_data['id'][i])
        x_val_text.append(whole_data['text'][i])
        x_val_w2v.append(whole_data['w2v'][i])
        x_val_tf_idf.append(whole_data['tf_idf'][i])
        x_val_pos.append(whole_data['pos'][i])
        y_val_has_tag.append(whole_data['y_has_tag'][i])
        y_val_tag_position.append(whole_data['y_tag_position'][i])
    else:
        x_train_w2v.append(whole_data['w2v'][i])
        x_train_tf_idf.append(whole_data['tf_idf'][i])
        x_train_pos.append(whole_data['pos'][i])
        y_train_has_tag.append(whole_data['y_has_tag'][i])
        y_train_tag_position.append(whole_data['y_tag_position'][i])
print len(x_train_w2v)        
x_train_w2v = np.array(x_train_w2v).astype('float32')
x_train_w2v = x_train_w2v.reshape(x_train_w2v.shape[0],1,max_len,word_embedding_size)
x_train_tf_idf = np.array(x_train_tf_idf).astype('float32')
x_train_tf_idf = x_train_tf_idf.reshape(x_train_tf_idf.shape[0],1,max_len,7)
x_train_pos = np.array(x_train_pos).astype('float32')
y_train_has_tag = np.array(y_train_has_tag).astype('float32')
y_train_tag_position = np.array(y_train_tag_position).astype('float32')

x_val_w2v = np.array(x_val_w2v).astype('float32')
x_val_w2v = x_val_w2v.reshape(x_val_w2v.shape[0],1,max_len,word_embedding_size)
x_val_tf_idf = np.array(x_val_tf_idf).astype('float32')
x_val_tf_idf = x_val_tf_idf.reshape(x_val_tf_idf.shape[0],1,max_len,7)
x_val_pos = np.array(x_val_pos).astype('float32')
y_val_has_tag = np.array(y_val_has_tag).astype('float32')
y_val_tag_position = np.array(y_val_tag_position).astype('float32')

print x_train_w2v.shape
print x_train_tf_idf.shape
print x_train_pos[0]
# model
model.fit( {'word_embed':x_train_w2v, 'tf_idf':x_train_tf_idf, 'pos':x_train_pos},
#model.fit( {'word_embed':x_train_w2v, 'tf_idf':x_train_tf_idf},
           {'has_tag':y_train_has_tag, 'tag_position':y_train_tag_position},
           validation_data=( {'word_embed':x_val_w2v, 'tf_idf':x_val_tf_idf, 'pos':x_val_pos},
                             {'has_tag':y_val_has_tag, 'tag_position':y_val_tag_position}),
           nb_epoch=100,
           batch_size=300)

# evaluate
ans = {}
for index, doc_id in enumerate(x_val_id):
    predict = model.predict([np.array([x_val_w2v[index]]),np.array([x_val_tf_idf[index]]),np.array([x_val_pos[index]])])
    if predict[0][0][0] > choose_th:
        words = word_tokenize(x_val_text[index])
        #for word_position in range(len(predict[1][0])):
        for word_position in range(len(words)):
            if predict[1][0][word_position] > acc_th:
                if doc_id not in ans:
                    ans[doc_id] = ""
                if word_position < len(words):
                    ans_word = words[word_position]
                if ans_word not in util.stopwords_set:
                    ans[doc_id] += ans_word + ' '

precision = []
recall =[]
f1_score = []
for index, tags in enumerate(document[val_class]['tags']):
    doc_id = document[val_class]['id'][index]
    pre_tag = ""
    if doc_id in ans:
        pre_tag = ans[doc_id]
    p,r,f = evaluate.f1_score(pre_tag, tags)
    precision.append(p)
    recall.append(r)
    f1_score.append(f)
print np.mean(precision), np.mean(recall), np.mean(f1_score)

outfile = open('cnn_result.csv','w')
for doc_id in ans:
    line = str(doc_id) + ',' + ans[doc_id] + '\n'
    outfile.write(line)

outfile.close()
