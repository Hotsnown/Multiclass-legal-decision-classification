import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

data = pd.read_csv('data/processed/output.csv')

data.loc[(data['formation'] == 'CHAMBRE_CIVILE_1') | (data['formation'] == 'CHAMBRE_CIVILE_2') | (data['formation'] == 'CHAMBRE_CIVILE_3'), 'LABEL'] = 0
data.loc[data['formation'] == 'CHAMBRE_CRIMINELLE', 'LABEL'] = 1
data.loc[data['formation'] == 'CHAMBRE_SOCIALE', 'LABEL'] = 2
data.loc[data['formation'] == 'CHAMBRE_COMMERCIALE', 'LABEL'] = 3
print(data['LABEL'][:10])

labels = to_categorical(data['LABEL'], num_classes=4)
print(labels[:10])
if 'CATEGORY' in data.keys():
    data.drop(['CATEGORY'], axis=1)

n_most_common_words = 8000
max_len = 130
tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['contenu'].values)
sequences = tokenizer.texts_to_sequences(data['contenu'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = pad_sequences(sequences, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=42)

epochs = 10
emb_dim = 128
batch_size = 256
labels[:2]

print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))

model = Sequential()
model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.7))
model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])

accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))