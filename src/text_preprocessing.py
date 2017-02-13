#-*- coding: utf8 -*-　
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import string
data_dir = '/home/hsienchin/transfer_learning/data/'

dataframes = {
    "cooking": pd.read_csv(data_dir + "cooking.csv"),
    "crypto": pd.read_csv(data_dir + "crypto.csv"),
    "robotics": pd.read_csv(data_dir + "robotics.csv"),
    "biology": pd.read_csv(data_dir + "biology.csv"),
    "travel": pd.read_csv(data_dir + "travel.csv"),
    "diy": pd.read_csv(data_dir + "diy.csv"),
}

print(dataframes["robotics"].iloc[1])

uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def stripTagsAndUris(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""

for df in dataframes.values():
    df["content"] = df["content"].map(stripTagsAndUris)

#print(dataframes["robotics"].iloc[1])

def removePunctuation(x):
    x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    # Removing (replacing with empty spaces actually) all the punctuations, without !?.
    return re.sub("["+string.punctuation+"]", " ", x) 
    #return re.sub("["+'"#$%&\'()*+,-/:;<=>@[\\]^_`{|}~'+"]", " ", x)

for df in dataframes.values():
    df["title"] = df["title"].map(removePunctuation)
    df["content"] = df["content"].map(removePunctuation)
'''
#print(dataframes["robotics"].iloc[1])
for name, df in dataframes.items():
    df.to_csv(data_dir + name + "_with_stop_words.csv", index=False)
def removeFinal(x):
    return re.sub("!?.", " ", x)
for df in dataframes.values():
    df["title"] = df["title"].map(removeFinal)
    df["content"] = df["content"].map(removeFinal)
'''

stops = set(stopwords.words("english"))
def removeStopwords(x):
    # Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)


for df in dataframes.values():
    df["title"] = df["title"].map(removeStopwords)
    df["content"] = df["content"].map(removeStopwords)

#print(dataframes["robotics"].iloc[1])

for name, df in dataframes.items():
    df.to_csv(data_dir + name + "_light.csv", index=False)
