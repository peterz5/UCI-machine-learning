from sklearn import svm, preprocessing, model_selection, metrics
import pandas as pd 
import numpy as np 

df = pd.read_csv('wine_data.csv')

features = df.drop('Class',1)
labels = df['Class']

x_train, x_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size = .2)
clf = svm.SVC(kernel='')
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

print(metrics.accuracy_score(predictions, y_test))