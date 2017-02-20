import nltk
import gensim,logging
import util
import Cython
data = util.load_data(data_dir='../data/', data_type='_with_stop_words_2.csv')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print "appending sentence..."
raw_sentences = []
for data_class in data:
    for title in data[data_class]['title']:
        raw_sentences.append(title)
    for content in data[data_class]['content']:
        for sentence in nltk.sent_tokenize(content):
            raw_sentences.append(sentence)
print "tokenize..."
sentences = map(nltk.word_tokenize, raw_sentences)
print "training model..."
model = gensim.models.Word2Vec(sentences, min_count=1, size=200, workers=4)
print "save model..."
model.save('my_word2vec_2.model')
