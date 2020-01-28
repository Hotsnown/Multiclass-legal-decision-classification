import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

data = pd.read_csv("data/processed/output.csv")

def build_labels():
    print("Building label targets...")
    labels = to_categorical(data["LABEL"], num_classes=4)
    return labels

def build_tokens():
    
    print("Building features...")
    n_most_common_words = 8000
    max_len = 130
    
    print("Building token features...")
    tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(data["contenu"].values)
    
    print("Building sequence features...")
    sequences = tokenizer.texts_to_sequences(data["contenu"].values)
    word_index = tokenizer.word_index
    
    print('Found %s unique tokens.' % len(word_index))
    X = pad_sequences(sequences, maxlen=max_len)
    return X