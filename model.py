from sklearn.svm import SVC
import numpy as np

def train_svm(X_train, y_train, kernel, params):
    svm = SVC(kernel=kernel, **params)
    svm.fit(X_train, y_train)
    return svm

def predict(model, X):
    return model.predict(X)