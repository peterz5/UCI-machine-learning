import pandas as pd
import numpy as np
from sklearn import preprocessing, naive_bayes
import pickle
from collections import Counter

def main():
	FEATURES = ['Age', 'Workclass', 'Final Weight', 'Education', 'Education-num', 'Marital-status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-country', 'Label']

	TRAIN = pd.read_csv('adult_train.csv', names=FEATURES)
	TEST = pd.read_csv('adult_test.csv', names=FEATURES)

	TRAIN=remove_garbage(TRAIN)
	numeritize(TRAIN)

	TEST=remove_garbage(TEST)
	numeritize(TEST)

	x_train = TRAIN.drop(['Final Weight', 'Label'], 1)
	y_train = TRAIN['Label']

	x_train = preprocessing.scale(x_train)

	clf = naive_bayes.GaussianNB()
	clf.fit(x_train, y_train)

	#with open('NB.pickle', 'wb') as f:
		#pickle.dump(clf, f)

	#pickle_in = open('support_vector.pickle', 'rb')
	#clf = pickle.load(pickle_in)

	x_test = TEST.drop(['Final Weight', 'Label'], 1)
	y_test = TEST['Label']

	x_test = preprocessing.scale(x_test)

	accuracy = clf.score(x_test, y_test)
	print(accuracy)


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


