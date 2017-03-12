import pandas as pd
import tf_idf
import random
import nltk
import string
from nltk import word_tokenize, sent_tokenize, pos_tag
import numpy as np
from collections import Counter
from nltk.corpus import stopwords

ww = []
for line in open('/home/hsienchin/transfer_learning_tag_detection/src/stop_word_list.txt'):
    ww.append(line.strip())
stopwords_set = set(ww)

def clean_sent(sent_str):
    return [ w for w in word_tokenize(str(sent_str)) if w not in string.punctuation ]

def clean_tag(tag_str):
    tag = []
    for t in tag_str.split():
        if '-' in t:
            for tt in t.split('-'):
                tag.append(tt)
        else:
            tag.append(t)
    return tag

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
    tokens = word_tokenize(sentence)
    word2count = Counter(tokens)
    return word2count

def find_position(word, sentence):
    words = clean_sent(sentence)
    if word in words:
        return words.index(word) + 1
    return 0

all_pos_tag = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']

def make_cnn_feature(data, word2vec_model_name, word_embedding_size=200, sent_size=10):
    import gensim
    word2vec = gensim.models.Word2Vec.load(word2vec_model_name)
    min_sent_len = 5
    feature_size = 7
    feature = {}
    feature['w2v'] = []
    feature['tf_idf'] = []
    feature['pos'] = []
    feature['y_has_tag'] = []
    feature['y_tag_position'] = []
    feature['text'] = []
    feature['id'] = []

    print "counting idf..."
    title_idf, content_idf = tf_idf.inverse_frequency(data, opt='smooth')

    print "counting words..."
    word2count = word_count(data)
    max_word_count = float(max(word2count.values()))
    isTitle = 1
    print "extract feature from titles..."
    for index, title in enumerate(data['title']):
        words = clean_sent(title)
        if len(words) < sent_size and len(words) > min_sent_len:
            tags = clean_tag(data['tags'][index])
            word_embedding = np.zeros((sent_size, word_embedding_size))
            word_tf_idf = np.zeros((sent_size, feature_size))
            pos_tag_seq = [0] * sent_size
            position = [0] * sent_size
            pos_tags = pos_tag(words)
            for i in range(0, min(sent_size,len(words))):
                if words[i] in word2vec:
                    word_embedding[i] = word2vec[ words[i] ]
                tf = tf_idf.term_frequency( words[i], title)
                isNotStopWord = 0 if words[i] in stopwords_set else 1
                word_tf_idf[i] = [ tf, title_idf[words[i]], tf*title_idf[words[i]], isTitle,
                                   isNotStopWord, word2count[words[i]]/max_word_count, i+1]
                if pos_tags[i][1] in all_pos_tag :
                    pos_tag_seq[i] = all_pos_tag.index(pos_tags[i][1]) + 1
            hasTag = False
            for tag in tags:
                if tag in words:
                    hasTag = True
                    position[words.index(tag)] = 1

            feature['w2v'].append(word_embedding)
            feature['tf_idf'].append(word_tf_idf)
            feature['pos'].append(pos_tag_seq)
            feature['y_has_tag'].append([1,0]) if hasTag else feature['y_has_tag'].append([0,1])
            feature['y_tag_position'].append(position)
            feature['text'].append(title)
            feature['id'].append(data['id'][index])

    print "extract feature from contents..."
    isTitle = 0
    for index, content in enumerate(data['content']):
        for sent in sent_tokenize(content):
            words = clean_sent(sent)
            if len(words) < sent_size and len(words) > min_sent_len:
                word_embedding = np.zeros((sent_size, word_embedding_size))
                word_tf_idf = np.zeros((sent_size, feature_size))
                pos_tag_seq = [0] * sent_size
                position = [0] * sent_size
                tags = clean_tag(data['tags'][index])
                pos_tags = pos_tag(words)
                for i in range(0, min(sent_size,len(words))):
                    if words[i] in word2vec:
                        word_embedding[i] = word2vec[ words[i] ]
                    tf = tf_idf.term_frequency( words[i], content)
                    isNotStopWord = 0 if words[i] in stopwords_set else 1
                    word_tf_idf[i] = [ tf, content_idf[words[i]], tf*content_idf[words[i]], isTitle,
                                       isNotStopWord, word2count[words[i]]/max_word_count, i+1]
                    if pos_tags[i][1] in all_pos_tag :
                        pos_tag_seq[i] = all_pos_tag.index(pos_tags[i][1]) + 1
                hasTag = False
                for tag in tags:
                    if tag in words:
                        hasTag = True
                        position[words.index(tag)] = 1

                feature['w2v'].append(word_embedding)
                feature['tf_idf'].append(word_tf_idf)
                feature['pos'].append(pos_tag_seq)
                feature['y_has_tag'].append([1,0]) if hasTag else feature['y_has_tag'].append([0,1])
                feature['y_tag_position'].append(position)
                feature['text'].append(sent)
                feature['id'].append(data['id'][index])
    feature['w2v'] = np.array(feature['w2v'])
    feature['tf_idf'] = np.array(feature['tf_idf'])
    feature['pos'] = np.array(feature['pos'])
    feature['y_has_tag'] = np.array(feature['y_has_tag'])
    feature['y_tag_position'] = np.array(feature['y_tag_position'])

    return feature
    

def make_feature(data_light, data_with_stop_words, negtive_rate=0.9):
    # for example: data_light = dataframe['cooking']
    x = []
    y = []
    title_idf, content_idf = tf_idf.inverse_frequency(data_light, opt='smooth')
    word2count = word_count(data_light)
    max_word_count = float(max(word2count.values()))
    for index, title in enumerate(data_light['title']):
        tags = clean_tag(data_light['tags'][index])
        for word in tf_idf.clean_string(title):
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


def n_word_feature(data, left=2, right=2, negtive_rate=0.9):
    from tf_idf import term_frequency
    # featrue [ tf, idf, tf*idf, isTitle, word_count, position+1 ]
    title_idf, content_idf = tf_idf.inverse_frequency(data, opt='smooth')
    word2count = word_count(data)
    max_word_count = float(max(word2count.values()))
    feature_size = 6
    ZERO = [0]*feature_size # 6 = feature size
    feature = {}
    feature['id'] = []
    feature['x'] = []
    feature['y'] = []
    feature['text'] = []
    for index, title in enumerate(data['title']):
        doc_id = data['id'][index]
        tags = clean_tag(data['tags'][index])
        words = clean_sent(title)
        temp_x, temp_y = [], []
        for index1, word in enumerate(words):
            tf = term_frequency(word, title)
            temp_x.append( [ tf, title_idf[word], tf*title_idf[word], 
                             1, word2count[word]/max_word_count, index1+1])
            temp_y.append( [1,0] ) if word in tags else temp_y.append( [0,1] )

        for index1, word in enumerate(words):
            if word not in stopwords_set:
                x, y = [], []
                for j in range(index1-left, index1+right+1):           
                    x.append(temp_x[j]) if j >= 0 and j < len(words) else x.append(ZERO)
                x = np.array(x).reshape( (left+1+right)*feature_size )
                y = temp_y[index1] 
                feature['id'].append(doc_id)
                feature['x'].append(x)
                feature['y'].append(y)
                feature['text'].append(word)
 
    for index, content in enumerate(data['content']):
        doc_id = data['id'][index]
        tags = clean_tag(data['tags'][index])
        sentenses = sent_tokenize(content)
        for sent in sentenses:
            words = clean_sent(sent)
            temp_x, temp_y = [], []
            for index1, word in enumerate(words):
                tf = term_frequency(word, content)
                temp_x.append( [ tf, content_idf[word], tf*content_idf[word], 
                                 1, word2count[word]/max_word_count, index1+1])
                temp_y.append( [1,0] ) if word in tags else temp_y.append( [0,1] )

            for index1, word in enumerate(words):
                if word not in stopwords_set:
                    x, y = [], []
                    for j in range(index1-left, index1+right+1):           
                        x.append(temp_x[j]) if j >= 0 and j < len(words) else x.append(ZERO)
                    x = np.array(x).reshape( (left+1+right)*feature_size )
                    y = temp_y[index1] 
                    feature['id'].append(doc_id)
                    feature['x'].append(x)
                    feature['y'].append(y)
                    feature['text'].append(word)

    feature['x'] = np.array(feature['x']).astype('float32')
    feature['y'] = np.array(feature['y']).astype('float32')
    return feature

