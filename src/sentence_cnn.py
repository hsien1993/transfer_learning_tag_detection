import util
import evaluate
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D, MaxPooling1D
import os

#feature = [word_embedding] + [tf, idf, tf*idf, isTitle, isStopWord, word_count, word_position, POS_tagging]
#word_embedding_size = 200
#pos_tagging_size = 10
#tf_idf_size = 6
word_embedding_model_name = 'my_word2vec'

# model
tf_idf_input = Input(shape=(tf_idf_size,), name='tf_idf_input')
word_input = Input(shape=

