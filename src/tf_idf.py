import pandas as pd
import math
import re
import nltk
import util
#def clean_string(s):
#    s = str(s)
#    if len(s) > 0:
#        return nltk.word_tokenize(s)
#    return s

def term_frequency(term, doc, opt='log'):
    doc = util.clean_sent(doc)
    if opt == 'simple':
        return 1.0 if term in doc else 0.0

    elif opt == 'log':
        t = 0
        for w in doc:
            if w == term:
                t += 1
        if t == 0:
            return 0.0
        return 1 + math.log(t)
    elif opt == 'aug':
        word = {}
        for w in doc:
            if w not in word:
                word[w] = 0
            word[w] += 1
        if term not in word:
            return 0.5
        return 0.5 + 0.5*(float(word[term])/max(word.values()))

def inverse_frequency(whole_doc, opt='smooth'):
    all_doc_num = 0
    title_word = {}
    content_word = {}
    for index, title in enumerate(whole_doc['title']):
        all_doc_num += 1
        doc_id = whole_doc['id'][index]
        for word in util.clean_sent(title):
            if word not in title_word:
                title_word[word] = []
            if doc_id not in title_word[word]:
                title_word[word].append(doc_id)
 
        for word in util.clean_sent(whole_doc['content'][index]):
            if word not in content_word:
                content_word[word] = []
            if doc_id not in content_word[word]:
                content_word[word].append(doc_id)

    title_idf = {}
    content_idf = {}
    if opt == 'smooth':
        for word in title_word:
            title_idf[word] = math.log(all_doc_num/(1+len(title_word[word])))
        for word in content_word:
            content_idf[word] = math.log(all_doc_num/(1+len(content_word[word])))
        return title_idf, content_idf

    elif opt == 'max':
        max_title_n = float(max([len(x) for x in title_word.values()]))
        max_content_n = float(max([len(x) for x in content_word.values()]))
        for word in title_word:
            title_idf[word] = math.log(max_title_n/(1+len(title_word[word])))
        for word in content_word:
            content_idf[word] = math.log(max_content_n/(1+len(content_word[word])))
        return title_idf, content_idf

    

if __name__ == '__main__':
    import evaluate
    import heapq
    import numpy
    data_dir = '/home/hsienchin/transfer_learning_tag_detection/data/'
    data_type = '_light.csv'
    title_only = True
    dataframes = {
        "cooking": pd.read_csv(data_dir + "cooking" + data_type),
        "crypto": pd.read_csv(data_dir + "crypto" + data_type ),
        "robotics": pd.read_csv(data_dir + "robotics" + data_type),
        "biology": pd.read_csv(data_dir + "biology" + data_type),
        "travel": pd.read_csv(data_dir + "travel" + data_type),
        "diy": pd.read_csv(data_dir + "diy" + data_type),
    }
    print "class, top_n, precision, recall, f1_score"
    #content_weight = 0.8
    for data_class in dataframes:
        title_idf, content_idf = inverse_frequency(dataframes[data_class], opt='smooth')
        for top_n in range(1,20):
            ans, f1, precision, recall = [],[],[],[]
            for index, title in enumerate(dataframes[data_class]['title']):
                predict_tags = ""
                content = dataframes[data_class]['content'][index]
                candidate = {}
                for word in util.clean_sent(title):
                    score = title_idf[word]*term_frequency(word, title)
                    if word in candidate:
                        if candidate[word] < score:
                            candidate[word] = score
                    else:
                        candidate[word] = score
                #for word in util.clean_sent(content):
                #    score = content_idf[word]*term_frequency(word, content)*content_weight
                #    if word in candidate:
                #        if candidate[word] < score:
                #            candidate[word] = score
                #    else:
                #        candidate[word] = score

                predict_tags = heapq.nlargest(top_n, candidate)
                p,r,f = evaluate.f1_score(" ".join(predict_tags), dataframes[data_class]['tags'][index])
                f1.append(f)
                precision.append(p)
                recall.append(r)  
            print  data_class, ',', top_n,',', numpy.mean(precision),',', numpy.mean(recall),',', numpy.mean(f1)
