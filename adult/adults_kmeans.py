import pandas as pd
import numpy as np
from sklearn import preprocessing, cluster
import pickle
from collections import Counter
from sklearn.metrics import accuracy_score

def main():
	FEATURES = ['Age', 'Workclass', 'Final Weight', 'Education', 'Education-num', 'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-country', 'Label']

	TRAIN = pd.read_csv('adult_train.csv', names=FEATURES)
	TEST = pd.read_csv('adult_test.csv', names=FEATURES)

	x = pd.concat((TRAIN, TEST), axis=0)
	original = pd.DataFrame.copy(x)

	x=remove_garbage(x)
	numeritize(x)

	y = x['Label']
	x = x.drop(['Final Weight', 'Label'], 1)

	x = preprocessing.scale(x)

	clf = cluster.KMeans(n_clusters=2, n_jobs=-1)
	clf.fit(x)

	#with open('KMeans.pickle', 'wb') as f:
		#pickle.dump(clf, f)

	pickle_in = open('KMeans.pickle', 'rb')
	clf = pickle.load(pickle_in)

	print(clf.cluster_centers_)
	print(clf.labels_)
	print(accuracy_score(clf.labels_, y))


def remove_garbage(df):
	df.replace(' ?', np.NaN, inplace=True)
	df.dropna(inplace=True)
	return df

def numeritize(df):

	columns = df.columns.values

	for col in columns:
		keys = {}

		def convert_to_num(val):
			return keys[val]

		datatype = df[col].dtype
		if datatype != np.int64 and datatype != np.float64:

			j = 0
			for i in df[col]:
				if not i in keys:
					keys[i] = j
					j+=1

			df[col] = df[col].apply(convert_to_num)

	return df

if __name__ == '__main__':
	main()


