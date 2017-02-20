import pandas as pd
import math
import re
import nltk
def clean_string(s):
    s = str(s)
    if len(s) > 0:
        return re.sub('\.|\!|\?',' ',s)
    return s

def term_frequency(term, doc, opt='log'):
    doc = clean_string(doc)
    if opt == 'simple':
        if term in doc.split():
            return 1.0
        return 0.0
    elif opt == 'log':
        t = 0
        for w in doc.split():
            if w == term:
                t += 1
        if t == 0:
            return 0.0
        return 1 + math.log(t)
    elif opt == 'aug':
        word = {}
        for w in doc.split():
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
        #title = clean_string(title)
        doc_id = whole_doc['id'][index]
        #for word in title.split():
        for word in nltk.word_tokenize(title):
            if word not in title_word:
                title_word[word] = []
            if doc_id not in title_word[word]:
                title_word[word].append(doc_id)
 
        #for word in clean_string(whole_doc['content'][index]).split():
        for word in nltk.word_tokenize(whole_doc['content'][index]):
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
    top_n = 5
    dataframes = {
        "cooking": pd.read_csv(data_dir + "cooking" + data_type),
        "crypto": pd.read_csv(data_dir + "crypto" + data_type ),
        "robotics": pd.read_csv(data_dir + "robotics" + data_type),
        "biology": pd.read_csv(data_dir + "biology" + data_type),
        "travel": pd.read_csv(data_dir + "travel" + data_type),
        "diy": pd.read_csv(data_dir + "diy" + data_type),
    }
    print "data_type, top_n, precision, recall, f1_score"
    for top_n in range(1,10):
        for data_class in dataframes:
            title_idf, content_idf = inverse_frequency(dataframes[data_class], opt='smooth')
            ans = []
            f1 = []
            precision = []
            recall = []
            for index, title in enumerate(dataframes[data_class]['title']):
                predict_tags = ""
                if title_only:
                    candidate = {}
                    for word in title.split():
                        # tf-idf scores
                        candidate[word] = title_idf[word]*term_frequency(word, title, opt='aug')
                    predict_tags = heapq.nlargest(top_n, candidate)
                    p,r,f = evaluate.f1_score(" ".join(predict_tags), dataframes[data_class]['tags'][index])
                    f1.append(f)
                    precision.append(p)
                    recall.append(r)  
            #print '-------------------------------------'
            print data_class, ',', top_n,',', numpy.mean(precision),',', numpy.mean(recall),',', numpy.mean(f1)
            #print 'precision: ', numpy.mean(precision)
            #print 'recall: ', numpy.mean(recall)
            #print 'f1_score: ', numpy.mean(f1)
