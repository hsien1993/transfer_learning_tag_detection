import pandas as pd
import tf_idf
import random
import nltk
from collections import Counter

def load_data(data_dir, data_type):
    # for example:  data_type = '_light.csv'
    #               data_dir = '/home/hsienchin/transfer_learning_tag_detection/data/'
    dataframes = {
        "cooking": pd.read_csv(data_dir + "cooking" + data_type),
        "crypto": pd.read_csv(data_dir + "crypto" + data_type ),
        "robotics": pd.read_csv(data_dir + "robotics" + data_type),
        "biology": pd.read_csv(data_dir + "biology" + data_type),
        "travel": pd.read_csv(data_dir + "travel" + data_type),
        "diy": pd.read_csv(data_dir + "diy" + data_type),
    }
    return dataframes

def word_count(doc):
    sentence = ' '.join([title for title in doc['title']]) + ' ' + ' '.join([content for content in doc['content']])
    tokens = nltk.word_tokenize(sentence)
    word2count = Counter(tokens)
    return word2count

def find_position(word, sentence):
    if word in nltk.word_tokenize(sentence):
        return sentence.index(word)
    return 0

def make_cnn_feature(data_light, data_with_stop_words, word_embedding_size=200, sent_size=10, word2vec_model_name):
    import gensim
    from nltk import word_tokenize, sent_tokenize, pos_tag
    import numpy as np
    try:
        word2vec_model = gensim.models.Word2Vec(word2vec_model_name)
        print "read word2vec model..."
    except:
        print "no pretrained word2vec model..."
    
    all_pos_tag = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
    word_embedding = []
    pos_tag_vec = []
    y_haveTag = []
    y_TagPosition = []
    for title in data_with_stop_words['title']:
        words = word_tokenize(title)
        temp = np.zeros((sent_size,word_embedding_size))
        if len(words) < sent_size:
            
            temp.append(
        

def make_feature(data_light, data_with_stop_words, negtive_rate=0.9):
    # for example: data_light = dataframe['cooking']
    x = []
    y = []
    title_idf, content_idf = tf_idf.inverse_frequency(data_light, opt='smooth')
    word2count = word_count(data_light)
    max_word_count = float(max(word2count.values()))
    for index, title in enumerate(data_light['title']):
        tags = data_light['tags'][index].split()

        for word in tf_idf.clean_string(title).split():
            tf = tf_idf.term_frequency(word, title)
            feature = [ tf, title_idf[word], tf*title_idf[word], 1, word2count[word]/max_word_count,
                        find_position(word, data_with_stop_words['title'][index]) ]
            if word in tags:
                x.append(feature)
                y.append([1, 0])
            else:
                if random.uniform(0,1) > negtive_rate:
                    x.append(feature)
                    y.append([0, 1])

        content = tf_idf.clean_string(data_light['content'][index])
        for word in content.split():
            tf = tf_idf.term_frequency(word, content)
            feature = [ tf, content_idf[word], tf*content_idf[word], 0, word2count[word]/max_word_count,
                        find_position(word, data_with_stop_words['content'][index]) ]
            if word in tags:
                x.append(feature)
                y.append([1, 0])
            else:
                if random.uniform(0,1) > negtive_rate:
                    x.append(feature)
                    y.append([0, 1])
    return x,y
