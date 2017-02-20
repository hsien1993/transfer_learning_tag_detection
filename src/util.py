import pandas as pd
import tf_idf
import random
import nltk
from collections import Counter
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
    tokens = nltk.word_tokenize(sentence)
    word2count = Counter(tokens)
    return word2count

def find_position(word, sentence):
    if word in nltk.word_tokenize(sentence):
        return sentence.index(word)
    return 0

def make_cnn_feature(data_light, data_with_stop_word, word2vec_model_name, word_embedding_size=200, sent_size=10):
    import gensim
    from nltk import word_tokenize, sent_tokenize, pos_tag
    import numpy as np
    word2vec_model = gensim.models.Word2Vec.load(word2vec_model_name)
    feature_size = 7
    all_pos_tag = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
    x_word_embedding = []
    x_pos_tag_seq = []
    x_tf_idf = []
    y_hasTag = []
    y_TagPosition = []
    print "counting idf..."
    title_idf, content_idf = tf_idf.inverse_frequency(data_with_stop_word, opt='smooth')
    print "counting words..."
    word2count = word_count(data_with_stop_word)
    max_word_count = float(max(word2count.values()))
    title_content = ['title', 'content']
    for col in title_content:
        print "doing "+col+"..."
        for index, text in enumerate(data_with_stop_word[col]):
            sentences = sent_tokenize(text)
            for sen in sentences:
                print "sen: " + sen
                words = word_tokenize(sen)
                words_light = word_tokenize(data_light[col][index])
                tags = clean_tag(data_with_stop_word['tags'][index])
                hasTag = False
                word_embedding = np.zeros((sent_size, word_embedding_size))
                word_tf_idf = np.zeros((sent_size, feature_size))
                pos_tag_seq = [0] * sent_size
                position = [0] * sent_size
                if len(words) < sent_size:
                    pos_tags = pos_tag(words)
                    for i in range(0, min(sent_size,len(words))):
                        word_embedding[i] = word2vec_model[words[i]]
                        tf = tf_idf.term_frequency(words[i], sen)
                        word_tf_idf[i][0] = tf                                  #tf
                        if col == 'title':                                      #idf
                            word_tf_idf[i][1] = title_idf[words[i]]             
                        else:
                            word_tf_idf[i][1] = content_idf[words[i]]
                        word_tf_idf[i][2] = tf*word_tf_idf[i][1]                #tf-idf
                        word_tf_idf[i][3] = 1 if col == 'title' else 0          #is title
                        word_tf_idf[i][4] = 1 if words[i] in words_light else 0  #is not stop word
                        word_tf_idf[i][5] = word2count[words[i]]/max_word_count #word count
                        word_tf_idf[i][6] = find_position(words[i], sen)
                        if pos_tags[i][1] in all_pos_tag :
                            pos_tag_seq[i] = all_pos_tag.index(pos_tags[i][1]) + 1
                    x_word_embedding.append(word_embedding)
                    x_tf_idf.append(word_tf_idf)
                    x_pos_tag_seq.append(pos_tag_seq)
                    print "pos: ", pos_tag_seq
                    for tag in tags:
                        if tag in words:
                            hasTag = True
                            position[words.index(tag)] = 1
                    if hasTag:
                        y_hasTag.append([1,0])
                    else:
                        y_hasTag.append([0,1])
                    y_TagPosition.append(position)
    return x_word_embedding, x_tf_idf, x_pos_tag_seq, y_hasTag, y_TagPosition
    

def make_feature(data_light, data_with_stop_words, negtive_rate=0.9):
    # for example: data_light = dataframe['cooking']
    x = []
    y = []
    title_idf, content_idf = tf_idf.inverse_frequency(data_light, opt='smooth')
    word2count = word_count(data_light)
    max_word_count = float(max(word2count.values()))
    for index, title in enumerate(data_light['title']):
        #tags = data_light['tags'][index]).split()
        tags = clean_tag(data_light['tags'][index])
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
