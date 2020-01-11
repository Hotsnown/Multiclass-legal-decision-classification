import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def show_loss_and_acc():
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def show_confusion_matrix(y_test, y_pred):
    y_pred = (y_pred > 0.5)
    conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=data.LABEL.values, yticklabels=data.LABEL.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("CONFUSION MATRIX \n", size=16);
    plt.show()

def show_metrics(y_test, y_pred):
    print('\t\t\t\tCLASSIFICATIION METRICS\n')
    print(metrics.classification_report(y_test, y_pred, ))