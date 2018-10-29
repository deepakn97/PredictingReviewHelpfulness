
# %%
import numpy as np
import os
import pandas as pd
import gzip
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk import FreqDist,ngrams,word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re

# %%

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# give me tokens, I will give non stop(word) tokens
def remove_stopwords(data):
    stop_words = Counter(stopwords.words('english'))
    ans = []
    for each in data:
        if(each not in stop_words.keys()):
            ans.append(each)
    return ans

# give me tokens, I will give you lemmatized tokens
def lemmatizer(data):
    lmtzr = WordNetLemmatizer()
    ans = []
    for each in data:
        ans.append(lmtzr.lemmatize(each))
    return ans

# same as above(replace lemmatize with stemmed)
def stemmer(data):
    ps = PorterStemmer()
    ans = []
    for each in data:
        ans.append(ps.stem(each))
    return ans

def cleanData(data):
    data = word_tokenize(data)
    data = lemmatizer(remove_stopwords(data))
    string = ' '.join(data)
    return np.asarray(data), string

def getData(path, category):
    df = getDF(path)
    # df = df[:2000] ## For practical purpose
    df.drop(columns=['reviewerID', 'reviewerName', 'reviewTime', 'unixReviewTime', 'asin', 'overall'], inplace = True)
    df['ProductType'] = category
    df = df[['helpful','reviewText','summary','ProductType']]
    data = df.values
    total = ""
    for i in range(data.shape[0]):
        data[i, 0] = round((data[i, 0][0]+1.0) / (data[i, 0][1] + 2.0),3) > 0.5
        string = (data[i,3] + " " + data[i, 2] + " " + data[i, 1]).lower()
        string = re.sub(r'[^\w\s]','',string)
        data[i, 2], data[i, 1] = cleanData(string)
        total += (data[i,1]+" ")

    vocab = np.unique(np.asarray(word_tokenize(total)))
    data = np.delete(data, [3], 1)

    return data, vocab
