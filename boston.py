import itertools

import pandas as pd 
import tensorflow as tf 

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio', 'medv']

FEATURES = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'tax', 'ptratio']

LABEL = 'medv'

def input_fn(data, n_epochs=None, shuffle=True):
	return tf.estimator.inputs.pandas_input_fn(
		x=pd.DataFrame({k: data[k].values for k in FEATURES}),
		y = pd.Series(data[LABEL].values), num_epochs=n_epochs, 
		shuffle=shuffle)

def main(unused_argv):
	training_set = pd.read_csv('boston_train.csv', skipinitialspace=True, skiprows=1, names=COLUMNS)
	test_set = pd.read_csv('boston_test.csv', skipinitialspace=True, skiprows=1, names=COLUMNS)
	prediction_set = pd.read_csv('boston_predict.csv', skipinitialspace=True, skiprows=1, names=COLUMNS)
	
	feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
	
	regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=[10, 10])
	regressor.train(input_fn=input_fn(training_set))
	ev = regressor.evaluate(input_fn = input_fn(test_set, num_epochs=1, shuffle=False))
	loss_score = ev['loss']

	print('loss: (0:f)'.format(loss_score))

	y = regressor.predict(input_fn=input_fn(prediction_set, num_epochs=1, shuffle=False))

	predictions = list(p['predictions'] for p in itertools.islice(y, 6))
	print('Predictions: {}'.format(str(predictions)))

if __name__ == '__main__':
	tf.app.run()
