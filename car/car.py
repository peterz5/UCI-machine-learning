
import pandas as pd 
import numpy as np 
from sklearn import preprocessing, model_selection, naive_bayes

def main():
	FEATURES = ['Buying', 'Maint', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Label'] 

	df = pd.read_csv('car_data.csv', names=FEATURES)
	str_to_num(df)

	x = df.drop('Label', 1)
	y = df['Label']

	x = preprocessing.scale(x)

	x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=.2)

	clf = naive_bayes.GaussianNB()
	clf.fit(x_train, y_train)

	print(clf.score(x_test, y_test))

def str_to_num(df):
	columns = df.columns.values

	for col in columns:
		keys = {}

		def lookup_num_(x):
			return keys[x]

		datatype = df[col].dtype
		if datatype != np.int64 and datatype != np.float64:
			k = 0

			for i in df[col]:
				if not i in keys:
					keys[i] = k
					k +=1

			df[col] = df[col].apply(lookup_num_)

	return df

if __name__ == '__main__':
	main()


