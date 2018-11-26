# %%
import source.utility as util
# import os
import numpy as np
import nltk
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split

# %%
data, vocab = util.getData(os.path.abspath('./data/reviews_Amazon_Instant_Video_5.json.gz'), 'Amazon_Instant_Video')

# %%

# Collecting all the documents
docs = np.asarray([row[1] for row in data])
label = np.asarray([row[0] for row in data])
label = list(label)
x = sum(label)
print (x)
# %%
# Converting the text to an array of integers
vocab_size = vocab.shape[0]
encoded_docs = [one_hot(d,vocab.shape[0]) for d in docs]

# %%
# Padding the Document
max_len = 0
for i,s in enumerate(encoded_docs):
    if(len(s)==1271): print(docs[i])
    max_len = max(max_len,len(s))

padded_docs = pad_sequences(encoded_docs, maxlen= max_len,padding = 'post')

X_train, X_test, y_train, y_test = train_test_split(padded_docs, label, test_size=0.3, random_state=0, stratify=label)

# %%
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length = max_len))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))


model1 = Sequential()
model1.add(Embedding(vocab_size, 32, input_length = max_len))
model1.add(LSTM(10, return_sequences=False))
model1.add(Dense(1,activation='sigmoid'))

# %%
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
# print (model.summary())
print (model1.summary())

# # %%
# model.fit(X_train, y_train, epochs = 50, verbose = 1)
#
# # %%
# loss ,accuracy = model.evaluate(X_test, y_test, verbose=1)
# print ('Accuracy: %f' % (accuracy*100))

# %%
model1.fit(X_train, y_train, epochs = 50, verbose = 1, batch_size=256)

# %%
loss ,accuracy = model1.evaluate(X_test, y_test, verbose=1)
print ('Accuracy: %f' % (accuracy*100))
