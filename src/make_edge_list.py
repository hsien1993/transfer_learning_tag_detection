from util import load_data, clean_sent, stopwords_set
from nltk import sent_tokenize
import numpy as np
from os.path import isfile
data_dir = '../data/'
data_type = '_with_stop_words_3.csv'
vocab_file = '../feature/word_vocab.npy'
bacov_file = '../feature/word_bacov.npy'
title_sent_file = '../feature/title_sent.npy'
content_sent_file = '../feature/content_sent.npy'
data_all = load_data(data_dir, data_type)

def isPair(w0, w1):
    if w0 in stopwords_set or w1 in stopwords_set:
        return False
    return True


def edge_list(sents, w2id, w_len=2, isWeight=False):
    g = {}
    max_count = 0
    for sent in sents:
        words = clean_sent(sent)
        for i in range(0, len(words)):
            for j in range( i+1, min(len(words), i+w_len+1) ):
                if isPair(words[i], words[j]):
                    w0, w1 = w2id[words[i]], w2id[words[j]]
                    pair = (w0, w1) if w0 < w1 else (w1, w0)
                    if pair not in g:
                        g[pair] = 0
                    g[pair] = g[pair] + 1 if isWeight else 1
                    if max_count < g[pair]:
                        max_count = g[pair]
    return g



if isfile(vocab_file) and isfile(title_sent_file) and isfile(content_sent_file) and isfile(bacov_file):
    print "reading exist file..."
    vocab = np.load(vocab_file).item()
    bacov = np.load(bacov_file).item()
    title_sent = np.load(title_sent_file).item()
    content_sent = np.load(content_sent_file).item()
else:
    print "making vocab and sentences list"
    vocab = {}
    bacov = {}
    title_sent = {}
    content_sent = {}
    index = 0
    for topic in data_all:
        if topic not in title_sent:
            title_sent[topic] = []
        if topic not in content_sent:
            content_sent[topic] = []
        for sent in data_all[topic]['title']:
            title_sent[topic].append(sent)
            for word in clean_sent(sent):
                if word not in stopwords_set:
                    if word not in vocab:
                        vocab[word] = index
                        bacov[index] = word
                        index += 1
        for content in data_all[topic]['content']:
            for sent in sent_tokenize(content):
                content_sent[topic].append(sent)
                for word in clean_sent(sent):
                    if word not in stopwords_set:
                        if word not in vocab:
                            vocab[word] = index
                            bacov[index] = word
                            index += 1

    np.save('../feature/word_vocab.npy', vocab)
    np.save('../feature/title_sent.npy', title_sent)
    np.save('../feature/content_sent.npy', content_sent)
    np.save('../feature/word_bacov.npy', bacov)
title_g = {}
content_g = {}
for topic in title_sent:
    title_g[topic] = edge_list(title_sent[topic], vocab, isWeight=True)
for topic in content_sent:
    content_g[topic] = edge_list(content_sent[topic], vocab, isWeight=True)

feat_dir = '../feature/'
all_feat_type = 'all_title_edge_list'
all_fname = feat_dir + all_feat_type
f2 = open(all_fname, 'w')
feat_type = '_title_edge_list'
for topic in title_g:
    fname = feat_dir + topic + feat_type
    f = open(fname, 'w')
    for pair in title_g[topic]:
        s = str(pair[0]) + ' ' + str(pair[1]) + ' ' + str(title_g[topic][pair]) + '\n'
        f.write(s)
        f2.write(s)
    f.close()
f2.close()

all_feat_type = 'all_content_edge_list'
all_fname = feat_dir + all_feat_type
f2 = open(all_fname, 'w')
feat_type = '_content_edge_list'
for topic in content_g:
    fname = feat_dir + topic + feat_type
    f = open(fname, 'w')
    for pair in content_g[topic]:
        s = str(pair[0]) + ' ' + str(pair[1]) + ' ' + str(content_g[topic][pair]) + '\n'
        f.write(s)
        f2.write(s)
    f.close()
f2.close()
