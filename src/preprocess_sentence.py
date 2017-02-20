import util
import preprocess_text
data_dir = '../data/'
data = util.load_data(data_dir,'.csv')

for df in data.values():
   df['content'] = df['content'].map(preprocess_text.stripTagsAndUris)

print data['cooking']['content'][10]
