import pyprind
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

#path to the directory containing the files
basepath = "aclImdb_v1/aclImdb/"

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)

#initialize an empty dataframe
df = pd.DataFrame()

for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()

df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

df = pd.read_csv('movie_data.csv', encoding='utf-8')
print('Displaying the dataframe\'s first 5 rows to ensure we read in the data correctly:')
print(df.head(5))
print('Shape of our DataFrame: ', df.shape)

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

nltk.download('stopwords')
stop = stopwords.words('english')
