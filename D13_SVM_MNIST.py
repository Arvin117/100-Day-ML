import numpy as np
import pandas as pd
from tools import load_mnist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, r2_score, classification_report

X_train, Y_train = load_mnist('./datasets/MNIST/raw', kind='train')
X_test, Y_test = load_mnist('./datasets/MNIST/raw', kind='t10k')
X_train, Y_train, X_test, Y_test = X_train[:10000, :], Y_train[:10000], X_test[:30, :], Y_test[:30]

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

svm = SVC(kernel='linear', random_state=0)
svm.fit(X_train, Y_train)
y_pred = svm.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
print(classification_report(Y_test, y_pred))
print(r2_score(Y_test, y_pred))
