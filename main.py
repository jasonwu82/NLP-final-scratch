import pandas as pd
import io_file
from model import BagWord
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


df = io_file.read_corpora_file("data/newsCorpora.shuffled.csv")
#df = io_file.read_corpora_file("data/corpora_small.csv")
print(df.head(1))


bag = BagWord()
# record word to dictionary
df.apply(lambda s: bag.add_line(s["title"]), axis=1)
#bag.add_line(df.loc[0, "title"])
#bag.add_line(df.loc[1, "title"])


df_train, df_dev, df_test = io_file.split_train_dev_test(df)

#df_test = df_train
x = bag.transform_lines(df_train.loc[0:1000, "title"].values)
# train
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(x, df_train.loc[0:1000, "category"])


#tmp = bag.transform_lines([df_train.loc[2, "title"]])
x_test = bag.transform_lines(df_dev.loc[0:1000, "title"].values)
y_true = df_dev.loc[0:1000, "category"]

res = classifier.predict(x_test)
print(res)

accuracy = np.sum(res == y_true) / len(res)
print(accuracy)
