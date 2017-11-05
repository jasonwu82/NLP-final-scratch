import pandas as pd
import io_file
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import time


if __name__ == "__main__":
    df = io_file.read_corpora_file("data/newsCorpora.shuffled.csv")

    df_train, df_dev, df_test = io_file.split_train_dev_test(df)
    train_all = True
    if train_all:
        data_train = df_train.loc[:, "title"]
        y_train = df_train.loc[0:, "category"]
    else:
        data_train = df_train.loc[:2000, "title"]
        y_train = df_train.loc[0:2000, "category"]
    data_test = df_dev.loc[:, "title"]
    y_test = df_dev.loc[:, "category"]

    vectorizer = HashingVectorizer(stop_words='english')
    x_train = vectorizer.transform(data_train)

    # build and train
    classifiers = []

    #classifiers.append((KNeighborsClassifier(n_neighbors=10), "KNN"))
    classifiers.append((LogisticRegression(), "Logistic Reg"))
    classifiers.append((LinearSVC(C=0.1), "linear SVM"))
    #classifiers.append((SVC(C=0.01), "SVM"))

    for clf, clf_name in classifiers:
        start_time = time.time()
        clf.fit(x_train, y_train)

        # test
        x_test = vectorizer.transform(data_test)

        y_test_pred = clf.predict(x_test)
        #print(y_test_pred)

        accuracy = np.sum(y_test_pred == y_test) / len(y_test)

        print("{1}: Training and predict using {0:2f} seconds".format(time.time() - start_time, clf_name))
        print("{0:3f}".format(accuracy))