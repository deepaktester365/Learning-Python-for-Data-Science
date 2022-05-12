from sklearn import tree, svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import time
import numpy as np
import pandas as pd
import sys
import matplotlib
from matplotlib import pyplot as plt

gender_dataset = pd.read_csv(
    './learning dataset/500_Person_Gender_Height_Weight_Index.csv')

X = np.asarray(gender_dataset[["Height", "Weight", "Index"]])
Y = np.asarray(gender_dataset["Gender"])


def tree_prediction(x_train, y_train, x_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    pred_equal = np.equal(prediction, y_test)
    return np.sum(pred_equal) / len(y_test)


def svm_prediction(x_train, y_train, x_test, y_test):
    clf = svm.SVC()
    clf = clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    pred_equal = np.equal(prediction, y_test)
    return np.sum(pred_equal) / len(y_test)


def log_res_prediction(x_train, y_train, x_test, y_test):
    clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    clf = clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    pred_equal = np.equal(prediction, y_test)
    return np.sum(pred_equal) / len(y_test)


def nearest_neighbors_prediction(x_train, y_train, x_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    prediction = clf.predict(x_test)
    pred_equal = np.equal(prediction, y_test)
    return np.sum(pred_equal) / len(y_test)


ratio_num = []
accuracy = {"Tree": [], "SVM": [],
            "Logistic Regression": [], "Nearest Neighbors": []}
for train_test_ratio_num in range(10, 490):
    ratio_num.append(train_test_ratio_num)
    X_train = X[0:train_test_ratio_num]
    Y_train = Y[0:train_test_ratio_num]

    X_test = X[train_test_ratio_num:500]
    Y_test = Y[train_test_ratio_num:500]

    start_time = time.time()
    accuracy["Tree"].append(tree_prediction(X_train, Y_train, X_test, Y_test))
    accuracy["SVM"].append(svm_prediction(X_train, Y_train, X_test, Y_test))
    accuracy["Logistic Regression"].append(
        log_res_prediction(X_train, Y_train, X_test, Y_test))
    accuracy["Nearest Neighbors"].append(
        log_res_prediction(X_train, Y_train, X_test, Y_test))

plt.plot(ratio_num, accuracy["Tree"], label="Tree")
plt.plot(ratio_num, accuracy["SVM"], label="SVM")
plt.plot(ratio_num, accuracy["Logistic Regression"],
         label="Logistic Regression")
# plt.plot(ratio_num, accuracy["Nearest Neighbors"], label="Nearest Neighbors")
plt.xlabel("Number of train_sample")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")
plt.show()
