# -*- coding: utf-8 -*-
import util
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

all_data = util.load_data('../data/','_light.csv')

l =' ' 
for data_class in all_data:
    l += ', ' + data_class
print l

l = '文章數量'
for data_class in all_data:
    n = len(all_data[data_class]['tags'])
    l += ', ' + str(n)
print l

l = '詞典大小'
for data_class in all_data:
    t = Tokenizer()
    t.fit_on_texts([str(content) for content in all_data[data_class]['content']])
    n = len(t.word_counts)
    l += ', ' + str(n)
print l


l = '關鍵詞數量'
for data_class in all_data:
    t = []
    for tag in all_data[data_class]['tags']:
        t.extend(tag.split())
    t = set(t)
    l += ', ' + str(len(t))
print l


