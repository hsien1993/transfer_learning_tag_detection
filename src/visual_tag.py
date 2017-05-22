import gensim
import util
from tsne import bh_sne
import numpy as np
from matplotlib import pyplot as plt
word2vec_model_name = '../model/my_word2vec.model'
word2vec = gensim.models.Word2Vec.load(word2vec_model_name)
all_data = util.load_data('../data/','_with_stop_words_3.csv')

all_class = [x for x in all_data]
data_tag = {}
i = 0
x = []
y = []
for data_class in all_data:
    data_tag[data_class] = []
    for tag in all_data[data_class]['tags']:
        tag = util.clean_tag(tag)
        data_tag[data_class].extend(tag)
    data_tag[data_class] = set(data_tag[data_class])

for data_class in data_tag:
    print data_class, i
    for tag in data_tag[data_class]:
        if tag in word2vec:
            x.append(word2vec[tag])
            y.append(i)
    i = i+1
x = np.array(x).astype('float64')
y = np.array(y)

vis_data = bh_sne(x)
vis_x = vis_data[:,0]
vis_y = vis_data[:,1]
plt.scatter(vis_x,vis_y,c=y,cmap=plt.cm.get_cmap("jet",6))
plt.colorbar(ticks=range(6))
plt.clim(-0.5,5.5)
plt.show()


