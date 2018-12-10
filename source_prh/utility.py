
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

# parse json file
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

# get pandas DataFrame from the json file
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

 # tokenize, lemmatize and remove stopwords from the given data
def cleanData(data):
    data = word_tokenize(data)
    data = lemmatizer(remove_stopwords(data))
    string = ' '.join(data)
    return data, string

# function to convert json to importable csv data file gor sql
def getDatatoCSV_sql(files):
    bigdata = pd.DataFrame(columns=['slno','product_id','product_type','reviewText','summary','reviewTime','overall','reviewerID','review_rating','ur'])
    for path, category in files:
        data = getDF(path)
        data = data.drop(columns=['reviewTime','reviewerName'])
        data = data.rename(columns={"asin" : "product_id","unixReviewTime":"reviewTime"})
        data['review_rating'] = 0.0
        data['ur'] = 0.0
        data['product_type'] = category
        data['slno'] = 0
        for i in range (data.shape[0]):
            if data['helpful'][i][0] + data['helpful'][i][1] == 0:
                    data.at[i,'review_rating'] = -1
            else:
                rr = round((data['helpful'][i][0]) / (data['helpful'][i][1]),3)
                if rr == 0.5 :
                        data.at[i,'review_rating'] = 1
                elif rr >0.5 :
                    data.at[i,'review_rating'] = 2
                else :
                    data.at[i,'review_rating'] = 0
            data.at[i,'slno'] = i
        data = data.drop(columns=['helpful'])
        data = data[['slno','product_id','product_type','reviewText','summary','reviewTime','overall','reviewerID','review_rating','ur']]
        bigdata=bigdata.append(data)
    bigdata.to_csv(os.path.abspath('./data/data_sql.csv'),index = False)

# get data from the final sql table
def getData(path):
    df = pd.read_csv(path)
    # df = df[:2000] ## For practical purpose
    df.drop(columns=['slno', 'product_id', 'reviewerID', 'reviewTime'], inplace = True)
    df = df[['review_rating', 'reviewText','summary', 'product_type', 'ur']]
    data = df.values
    total = ""
    for i in range(data.shape[0]):
        # rev.append(data[i,0][1])
        # give four labels, -1,0,1,2.
        # -1: no reviews
        # 0: not helpful
        # 1: neutral
        # 2: helpful

        # append product_type, summary to review text
        string = (data[i, 3] + " " + data[i, 2] + " " + data[i, 1]).lower()
        string = re.sub(r'[^\w\s]','',string)
        data[i, 1], data[i, 2] = cleanData(string)
        total += (data[i,2]+" ")

    # generate vocabulary to use for embedding
    vocab = np.unique(np.asarray(word_tokenize(total)))
    return data, vocab
