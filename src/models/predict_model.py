from sklearn.model_selection import train_test_split
from src.features.build_features import *
from src.visualization.visualize import show_confusion_matrix
from src.visualization.visualize import show_metrics

import pickle

labels = build_labels()
X = build_tokens()

SEED = 0
X_train, X_test, y_train, y_test = train_test_split(X , labels, test_size=0.25, random_state=SEED)

print("Train: N = {0} records".format(len(X_train)))
print("Test:  N = {0} records".format(len(X_test)))

with open("models/saved_model.pickle", "rb") as file: # Use file to refer to the file object
    
    print('Reading models/saved_model.pickle...')
    model_from_pickle = pickle.load(file)

    print('Predicting...')
    y_pred = model_from_pickle.predict(X_test)
    
    show_metrics(y_test, y_pred)
    show_confusion_matrix(y_test, y_pred)
