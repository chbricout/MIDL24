from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

def analyse_classif(label, pred):
    cf_matrix=confusion_matrix(label, pred)
    sb.heatmap(cf_matrix, annot=True)
    plt.xlabel(f"Predicted Label \nAccuracy : {accuracy_score(label, pred)}")
    plt.ylabel("True Label")
    plt.title(f"")

    cm = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    return cm.diagonal()