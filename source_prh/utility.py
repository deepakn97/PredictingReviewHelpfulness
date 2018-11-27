
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

# rev = []
def getData(path, category):
    df = getDF(path)
    df = df[:2000] ## For practical purpose
    df.drop(columns=['reviewerID', 'reviewerName', 'reviewTime', 'unixReviewTime', 'asin', 'overall'], inplace = True)
    df['ProductType'] = category
    df = df[['helpful','reviewText','summary','ProductType']]
    data = df.values
    total = ""
    for i in range(data.shape[0]):
        # rev.append(data[i,0][1])
        # give four labels, -1,0,1,2.
        # -1: no reviews
        # 0: not helpful
        # 1: neutral
        # 2: helpful

        lab = round((data[i, 0][0]+1.0) / (data[i, 0][1] + 2.0),3)
        if(data[i,0] == 0 and data[i,1] == 0): data[i,0] = -1
        elif(lab == 0.5): data[i,0] = 1
        elif(lab<0.5): data[i, 0] = 0
        else: data[i, 0] = 2

        string = (data[i,3] + " " + data[i, 2] + " " + data[i, 1]).lower()
        string = re.sub(r'[^\w\s]','',string)
        data[i, 2], data[i, 1] = cleanData(string)
        total += (data[i,1]+" ")

    vocab = np.unique(np.asarray(word_tokenize(total)))
    data = np.delete(data, [3], 1)
    return data, vocab

def getDatatoCSV_sql(path,category):
    data = getDF(path)
    data = data.drop(columns=['reviewTime','reviewerName'])
    data = data.rename(columns={"asin" : "product_id","unixReviewTime":"reviewTime"})
    data['review_rating'] = 0.0
    data['ur'] = 0.0
    for i in range (data.shape[0]):
        if data['helpful'][i][0] + data['helpful'][i][1] == 0:
                data.at[i,'review_rating'] = 0.500
        else :
            data.at[i,'review_rating'] = round((data['helpful'][i][0]) / (data['helpful'][i][1]),3)
    data = data.drop(columns=['helpful'])
    data.to_csv(os.path.abspath('./data/reviews_Amazon_Instant_Video_5.csv'))