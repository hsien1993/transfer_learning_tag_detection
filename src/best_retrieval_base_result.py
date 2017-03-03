import evaluate
import util
import numpy as np
data_light = util.load_data('/home/hsienchin/transfer_learning_tag_detection/data/','_light.csv')
print 'class, precision, recall, f1'
for data_class in data_light:
    precision = []
    recall = []
    f1_score = []
    for index, title in enumerate(data_light[data_class]['title']):
        raw_tags = data_light[data_class]['tags'][index]
        tags = raw_tags.split()
        sentence = str(title) + ' ' + str(data_light[data_class]['content'][index])
        ans = ' '.join(set(sentence.split())&set(tags))
        p,r,f = evaluate.f1_score(ans,raw_tags)
        precision.append(p)
        recall.append(r)
        f1_score.append(f)
    print data_class, ',', np.mean(precision), ',', np.mean(recall), ',', np.mean(f1_score)
