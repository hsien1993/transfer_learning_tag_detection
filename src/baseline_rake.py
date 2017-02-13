import RAKE
import pandas as pd
import evaluate 
data_dir = '/home/hsienchin/transfer_learning_tag_detection/data/'
with_out_stopwords = '_light.csv'
isBigram = 0
title_only = 0
dataframes = {
    "cooking": pd.read_csv(data_dir + "cooking" + with_out_stopwords),
    "crypto": pd.read_csv(data_dir + "crypto" + with_out_stopwords),
    "robotics": pd.read_csv(data_dir + "robotics" + with_out_stopwords),
    "biology": pd.read_csv(data_dir + "biology" + with_out_stopwords),
    "travel": pd.read_csv(data_dir + "travel" + with_out_stopwords),
    "diy": pd.read_csv(data_dir + "diy" + with_out_stopwords),
}

rake_object = RAKE.Rake('/home/hsienchin/python-rake/stoplists/SmartStoplist.txt')
for df in dataframes:
    ans = []
    precision = []
    recall = []
    f1 = []

    for index, title in enumerate(dataframes[df]['title']):
        if title_only:
            s = ""
            sen = title
            result_rake = rake_object.run(sen)
            if len(result_rake) > 0:
                s = result_rake[0][0]
            ans.append(str(s))
        else:
            content = str(dataframes[df]['content'][index])
            sen = title + content
            s = ""
            for a in rake_object.run(sen):
                s += a[0]
            ans.append(str(s))

    for index, tags in enumerate(dataframes[df]['tags']):
        p,r,f = evaluate.f1_score(ans[index],tags,isBigram)
        precision.append(p)
        recall.append(r)
        f1.append(f)
    print '------------------------------------------------'
    print df
    print 'precision: ', sum(precision)/float(len(precision))
    print 'recall: ', sum(recall)/float(len(recall))
    print 'f1: ', sum(f1)/float(len(f1))
