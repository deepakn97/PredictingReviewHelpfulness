# %%
import source.utility as util
import os
import numpy as np
import nltk
import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

# %%
data = util.getData(os.path.abspath('../PredictingReviewHelpfulness/Data/reviews_Amazon_Instant_Video_5.json.gz'), 'Amazon_Instant_Video')
# %%
print (data.shape)

# %%

vocab = np.asarray([])
c = 0
for row in data:
    if row[0]>0.5 : c+=1
    row[0] = row[0]>0.5
    vocab = np.append(vocab,row[2])
vocab = np.unique(vocab)

print (c)

# %% Collecting all the documents
docs = np.asarray([row[1] for row in data])
label = np.asarray([row[0] for row in data])
# %% Converting the text to an array of integers
vocab_size = vocab.shape[0]
encoded_docs = [one_hot(d,vocab.shape[0]) for d in docs]

# %% Padding the Document
max_len = 0
for str in docs:
    max_len = max(max_len,len(str))

padded_docs =pad_sequences(encoded_docs,maxlen= max_len,padding = 'post')


# %%
model = Sequential()
model.add(Embedding(vocab_size,32, input_length = max_len))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))

# %%
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
print (model.summary())

# %%
model.fit(padded_docs,label,epochs = 50,verbose = 1)

# %%
loss ,accuracy = model.evaluate(padded_docs,label, verbose=1)
print ('Accuracy: %f' % (accuracy*100))
