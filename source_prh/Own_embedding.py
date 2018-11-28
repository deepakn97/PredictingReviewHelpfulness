
# %%
import source_prh.utility as util
import os
import numpy as np
import nltk
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# %%
data, vocab = util.getData('./data/product_data.csv', 'Amazon_Instant_Video')
useful_data = [row for row in data if row[0] != -1]
noreview_data = [row for row in data if row[0] == -1]
# %%

# Collecting all the documents
docs_useful = list(np.asarray([row[2] for row in useful_data]))
label_useful = list(np.asarray([row[0] for row in useful_data]))

docs_noreview = list(np.asarray([row[2] for row in noreview_data]))
label_noreview = list(np.asarray([row[0] for row in noreview_data]))

# %%
print(docs_useful[0])
# x = sum(label)
# print (x)
docs = docs_useful+docs_noreview
labels = label_useful+label_noreview

# %%
# Converting the text to an array of integers
vocab_size = vocab.shape[0]
encoded_docs = [one_hot(d,vocab.shape[0]) for d in docs]

# %%
# Padding the Document
max_len = 0
for i,s in enumerate(encoded_docs):
    # if(len(s)==1271): print(docs[i])
    max_len = max(max_len,len(s))

padded_docs = pad_sequences(encoded_docs, maxlen= max_len,padding = 'post')
padded_useful = np.asarray(padded_docs[:len(docs_useful)])
padded_noreview = np.asarray(padded_docs[len(docs_useful):])
X_train, X_test, y_train, y_test = train_test_split(padded_useful, label_useful, test_size=0.3, random_state=0, stratify=label_useful)

# %%
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length = max_len))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))


model1 = Sequential()
model1.add(Embedding(vocab_size, 32, input_length = max_len))
model1.add(LSTM(256, dropout=0.5, return_sequences=True, activation='sigmoid'))
model1.add(LSTM(256, dropout=0.5, return_sequences=False, activation='sigmoid'))
model1.add(Dense(3,activation='softmax'))

# %%
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
# print (model.summary())
print (model1.summary())

# %%
# model.fit(X_train, y_train, epochs = 50, verbose = 1)
#
# # %%
# loss ,accuracy = model.evaluate(X_test, y_test, verbose=1)
# print ('Accuracy: %f' % (accuracy*100))
# %%

y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# %%
model1.fit(X_train, y_train, epochs = 6, verbose = 1, batch_size=256)

# %%
loss ,accuracy = model1.evaluate(X_test, y_test, verbose=1)
print ('Accuracy: %f' % (accuracy*100))
