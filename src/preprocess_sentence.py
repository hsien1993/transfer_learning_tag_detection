import util
import preprocess_text

def removePunctuation(x):
    x = x.lower()
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    return x

data_dir = '../data/'
data = util.load_data(data_dir,'.csv')

for df in data.values():
    df['content'] = df['content'].map(preprocess_text.stripTagsAndUris)

print data['cooking']['content'][10]

for df in data.values():
    data['content'] = df['content'].map(removePunctuation)
