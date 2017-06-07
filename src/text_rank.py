import nltk
from util import stopwords_set, clean_sent
from collections import Counter, defaultdict

def coocurance(text,windows=2):
    word_lst = [e for e in clean_sent(text) if e not in stopwords_set]
    #print '/'.join(word_lst)
    data = defaultdict(Counter)
    for i,word in enumerate(word_lst):
        indexStart = i - windows
        indexEnd = i + windows
        if indexStart < 0:
            temp = Counter(word_lst[:windows+1+i])
            temp.pop(word)
            data[word] += temp       
            # print word
        elif indexStart>=0 and indexEnd<=len(word_lst):
            temp = Counter(word_lst[i-windows:i+windows+1])
            temp.pop(word)
            data[word] += temp
        else:
            temp = Counter(word_lst[i-windows:])
            temp.pop(word)
            data[word]+=temp
            # print word
    return data

def textRank(graph,d=0.85,kw_num=10,with_weight=False):
    TR = defaultdict(float,[(_word,1.) for _word,cooc in graph.items()]) # TextRank graph with default 1

    TR_prev = TR.copy()
    err = 1e-4
    error = 1
    iter_no = 100
    index = 1
    while (iter_no >index and  error > err):
        error = 0
        TR_prev = TR.copy()
        for word,cooc in graph.items():
            temp=0
            for link_word,weight in cooc.items():
                temp += d*TR[link_word]*weight/sum(graph[link_word].values())
                # print 'temp:',temp
            TR[word] = 1 -d + temp
            #print 'word:%s,TR:%.2f'%(word.encode('utf8'),TR[word])
            # print 'TR[{}]:{}'.format(word.encode('utf8'),TR[word])
            # print '----'
        error += (TR[word] - TR_prev[word])**2
        #print '-'*40
        # print 'keywords finding...iter_no:{},\terror:{}'.format(index,error)
        index += 1
    if with_weight:
        kw = sorted(TR.iteritems(),key=lambda (k,v):(v,k),reverse=True)
        kw = [(k,v/max(zip(*kw)[1])) for k,v in kw ][:kw_num]
    else:
        kw = [word for word,weight in sorted(TR.iteritems(),key=lambda (k,v):(v,k),reverse=True)[:kw_num]]
    return kw
if __name__ == '__main__':
    import util
    data_dir = '../data/'
    data_type = '_light.csv'
    all_data = util.load_data(data_dir, data_type)
    all_num = [1,2,3,4,5,6,7]
    for data_class in all_data:
        for num in all_num:
            f = open('../result/'+data_class+'_'+str(num)+'_text_rank_result.csv','w')
            for index,content in enumerate(all_data[data_class]['content']):
                sent_graph = coocurance(content)
                if len(sent_graph) > 0:
                    kw = textRank(sent_graph, kw_num=num)
                    #print all_data[data_class]['id'][index], ',', kw
                    f.write(str(all_data[data_class]['id'][index])+','+' '.join(kw)+','+all_data[data_class]['tags'][index]+'\n')
            f.close()
