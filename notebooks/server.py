from flask import Flask, jsonify
from flask import request
from sklearn.externals import joblib
import pandas as pd
import request
from flask import *

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

def build_tokens(data):
    
    print("Building features...")
    n_most_common_words = 8000
    max_len = 130
    
    print("Building token features...")
    tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(data)
    
    print("Building sequence features...")
    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    
    print('Found %s unique tokens.' % len(word_index))
    X = pad_sequences(sequences, maxlen=max_len)
    return X


app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
     session = tf.Session(config=config)
     json_ = request.get_json()
     query = build_tokens(json_["data"])
     prediction = clf.predict(query)
     return jsonify({'prediction': prediction.tolist()})
if __name__ == '__main__':
     clf = joblib.load('models/saved_model.pickle')
     app.run(port=8080, threaded=False)