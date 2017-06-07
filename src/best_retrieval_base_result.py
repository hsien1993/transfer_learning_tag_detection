import evaluate
import util
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
data_light = util.load_data('/home/hsienchin/transfer_learning_tag_detection/data/','_light.csv')
print 'class, precision, recall, f1'
for data_class in data_light:
    precision = []
    recall = []
    f1_score = []
    for index, title in enumerate(data_light[data_class]['title']):
        tags = set(text_to_word_sequence(data_light[data_class]['tags'][index]))
        #tags = raw_tags.split()
        sentence = text_to_word_sequence(str(title) + ' ' + str(data_light[data_class]['content'][index]))
        ans = ' '.join([x for x in set(sentence) if x in tags])
        p,r,f = evaluate.f1_score(ans,' '.join(tags))
        precision.append(p)
        recall.append(r)
        f1_score.append(f)
    print data_class, ',', np.mean(precision), ',', np.mean(recall), ',', np.mean(f1_score)
