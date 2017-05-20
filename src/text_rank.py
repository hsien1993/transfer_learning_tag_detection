import nltk
from util import stopwords_set, clean_sent
from collections import Counter, defaultdict

def coocurance(text,windows=3):
    word_lst = [e for e in clean_sent(text) if e not in stopwords_set]
    print '/'.join(word_lst)
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

def textRank(graph,d=0.85,kw_num=3,with_weight=False):
    TR = defaultdict(float,[(word,1.) for word,cooc in graph.items()]) # TextRank graph with default 1

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
            print 'word:%s,TR:%.2f'%(word.encode('utf8'),TR[word])

            # print 'TR[{}]:{}'.format(word.encode('utf8'),TR[word])

            # print '----'

        error += (TR[word] - TR_prev[word])**2
        print '-'*40
        # print 'keywords finding...iter_no:{},\terror:{}'.format(index,error)

        index += 1
    if with_weight:
        kw = sorted(TR.iteritems(),key=lambda (k,v):(v,k),reverse=True)
        kw = [(k,v/max(zip(*kw)[1])) for k,v in kw ][:kw_num]
    else:
        kw = [word for word,weight in sorted(TR.iteritems(),key=lambda (k,v):(v,k),reverse=True)[:kw_num]]
    return kw
s = 'one definition questions also one interests personally find guide take safely amazon jungle love explore amazon would attempt without guide least first time prefer guide going ambush anything p edit want go anywhere touristy start end points open trip take places likely see travellers tourists definitely require good guide order safe'
#s = "this was one of our definition questions  but also one that interests me personally  how can i find a guide that will take me safely through the amazon jungle? i'd love to explore the amazon but would not attempt it without a guide  at least not the first time  and i'd prefer a guide that wasn't going to ambush me or anything  predit  i don't want to go anywhere  touristy    start and end points are open  but the trip should take me places where i am not likely to see other travellers   tourists and where i will definitely require a good guide in order to be safe"
data = coocurance(s)
kw = textRank(data)
print kw
