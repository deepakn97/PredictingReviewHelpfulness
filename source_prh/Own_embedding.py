
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


# %%
# Converting the text to an array of integers
vocab_size = vocab.shape[0]
for rows in data:
    rows[2] = one_hot(rows[2], vocab_size)
# %%
# Padding the Document

max_len = 0
for i,s in enumerate(data[:,2]):
    # if(len(s)==1271): print(docs[i])
    max_len = max(max_len,len(s))

padded_docs = pad_sequences(data[:,2], maxlen= max_len,padding = 'post')
for i, rows in enumerate(data):
    rows[2] = padded_docs[i]

# %%
useful_data = np.asarray([row for row in data if row[0] != -1])
noreview_data = np.asarray([row for row in data if row[0] == -1])

X_train, X_test, y_train, y_test = train_test_split(useful_data[:, 1:], useful_data[:, 0], test_size=0.3, random_state=0, stratify=useful_data[:, 0])


# %%
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length = max_len))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))


model1 = Sequential()
model1.add(Embedding(vocab_size, 32, input_length = max_len))
model1.add(LSTM(25, dropout=0.5, return_sequences=True, activation='sigmoid'))
model1.add(LSTM(25, dropout=0.5, return_sequences=False, activation='sigmoid'))
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
docs_train = np.asarray([X_train[i, 1] for i in range(len(X_train))])
docs_test = np.asarray([X_test[i, 1] for i in range(len(X_test))])

# %%
# %%
model1.fit(docs_train, y_train, validation_split=0.1, epochs = 6, verbose = 1, batch_size=256)

# %%
train_text_probs = model1.predict(docs_train, batch_size= 32, verbose = 1)
train_test_probs= np.append(train_test_probs, X_train[:, 2], axis = 1)

# %%
from  sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

n = train_text_probs.shape[0]
features = 3
X,y  = make_classification(n_samples = n, n_features = features, shuffle = True)
classifier = RandomForestClassifier(n_estimators = 4,max_depth = 3)

classifier.fir(X,y)
classifier.predict

# %%
loss ,accuracy = model1.evaluate(X_test, y_test, verbose=1)
print ('Accuracy: %f' % (accuracy*100))
