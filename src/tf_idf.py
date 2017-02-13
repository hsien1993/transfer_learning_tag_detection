import pandas as pd
def clean_string(s):
    return s.strip('.').strip('!').strip('?')

def term_frequency(doc):
    # simple tf
    tf = {}
    word_count = 0
    for title in doc['title']:
        title = clean_string(title)
        words = tilte.split(' ')
        for word in words:
            if word not in tf:
                tf[word] = 0
            tf[word] += 1
            word_count += 1
    for content in doc['content']:
        content = clean_string(content)
        words = tilte.split(' ')
        for word in words:
            if word not in tf:
                tf[word] = 0
            tf[word] += 1
            word_count += 1
    for word in tf:
        tf[word] = tf[word]/wordcount
    return tf

def inverse_frequency(doc):


if __name__ == 'main':
    data_dir = '/home/hsienchin/transfer_learning_tag_detection/data/'
    data_type = '_light.csv'
    dataframes = {
        "cooking": pd.read_csv(data_dir + "cooking" + data_type),
        "crypto": pd.read_csv(data_dir + "crypto" + data_type ),
        "robotics": pd.read_csv(data_dir + "robotics" + data_type),
        "biology": pd.read_csv(data_dir + "biology" + data_type),
        "travel": pd.read_csv(data_dir + "travel" + data_type),
        "diy": pd.read_csv(data_dir + "diy" + data_type),
    }
    
