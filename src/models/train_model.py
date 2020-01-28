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
import pickle 

from src.features.build_features import *
from src.visualization.visualize import show_loss_and_acc

labels = build_labels()
X = build_tokens()

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=SEED)

epochs = 3
emb_dim = 128
batch_size = 256
labels[:2]
n_most_common_words = 8000

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

model_to_pickle = open("models/saved_model.pickle","wb")
pickle.dump(model, model_to_pickle)
model_to_pickle.close()

model_from_pickle = pickle.load(open("models/saved_model.pickle", "rb"))

model.save(f"models/legal-case-classifierV1.h5")