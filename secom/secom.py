import pandas as pd
import numpy as np 
from sklearn import model_selection, ensemble, preprocessing
import pickle

FEATURES = ['Feature ' + str(i) for i in range(590)]

x = pd.read_csv('secom_features.csv', names = FEATURES)
y = pd.read_csv('secom_labels.csv', names = ['Labels'])

for col in x.columns.values:
	x[col].replace(np.nan, pd.Series.mean(x[col]), inplace=True)

x = preprocessing.scale(x)
y = np.asarray(y)
y = y[:, 0]

#x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=.1)

kf = model_selection.KFold(n_splits=10)

for train_index, test_index in kf.split(x):
	x_train, x_test = x[train_index], x[test_index]
	y_train, y_test = y[train_index], y[test_index]

clf = ensemble.RandomForestClassifier(n_jobs = -1)
clf.fit(x_train, y_train)

#with open('RandomForest.pickle', 'wb') as f:
#	pickle.dump(clf, f)

pickle_in =  open('RandomForest.pickle', 'rb')
clf = pickle.load(pickle_in)

print(clf.score(x_test, y_test))
feats = clf.feature_importances_
print(feats)
print(FEATURES[np.argmax(feats)], ', ', np.max(feats))