
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# %%

# Loading the dataset
data, vocab = util.getData('./data/product_data.csv')

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

padded_docs = pad_sequences(data[:, 2], maxlen= max_len, padding = 'post')
for i, rows in enumerate(data):
    rows[2] = padded_docs[i]

# %%
# Seperate the data with no reviews
useful_data = np.asarray([row for row in data if row[0] != -1])
noreview_data = np.asarray([row for row in data if row[0] == -1])

X_train, X_test, y_train, y_test = train_test_split(useful_data[:, 1:], useful_data[:, 0], test_size=0.3, random_state=0, stratify=useful_data[:, 0])


# %%

# Define models

# Simple ANN model
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length = max_len))
model.add(Flatten())
model.add(Dense(3,activation='softmax'))


# Language Model
model1 = Sequential()
model1.add(Embedding(vocab_size, 32, input_length = max_len))
model1.add(LSTM(12, dropout=0.4, return_sequences=True, activation='tanh'))
model1.add(LSTM(12, dropout=0.4, return_sequences=False, activation='tanh'))
model1.add(Dense(3,activation='softmax'))

# %%
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
# model1.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])
# print (model.summary())
# print (model1.summary())

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
# print(docs_train.shape)
# %%
history = model.fit(docs_train, y_train, validation_split=0.3, epochs = 50, verbose = 1, batch_size=64)

# %%
# plotting model loss and accuracy
import matplotlib.pyplot as plt
# print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# %%
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# %%
train_text_probs = model1.predict(docs_train, batch_size= 32, verbose = 1)
test_text_probs = model1.predict(docs_test, batch_size= 32, verbose = 1)

# %%

# user rating from the data
ur_train = np.asarray([X_train[i, 3] for i in range(len(X_train))]).reshape(X_train.shape[0], 1)
ur_test = np.asarray([X_test[i, 3] for i in range(len(X_test))]).reshape(X_test.shape[0], 1)

# %%

# augmented data for ML model, LSTM output + user rating
X_train_mlmodel = np.append(train_text_probs, ur_train, axis = 1)
X_test_mlmodel = np.append(test_text_probs, ur_test, axis = 1)

# %%
# defining ML model(any other model will also work)
classifier = RandomForestClassifier(n_estimators = 4,max_depth = 3)

classifier.fit(X_train_mlmodel,y_train)

# %%
classifier.score(X_test_mlmodel, y_test)
