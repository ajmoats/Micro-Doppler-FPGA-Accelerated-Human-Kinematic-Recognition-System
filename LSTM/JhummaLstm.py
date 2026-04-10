"""
Jeff Craley 
February 10, 2016

Train an LSTM for each of the 5 data folds.
"""

from __future__ import absolute_import
from __future__ import print_function

import numpy
import scipy.io


from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
import pickle
import json
import csv
import h5py
import string

# Note: theano is a discontinued module, update wrappers/backend as necessary
# from theano import function, config, shared, sandbox
# import theano.sandbox.cuda.basic_ops
import os
os.environ['KERAS_BACKEND'] = 'tensorflow' # replaces theano
import matplotlib.pyplot as plt

import LSTM.data_loading as data_loading


def save_model(model, filename):
	"""
	Save the model as a json string
	Save the weights as a .h5 file
	"""
	json_string = model.to_json()
	open('../models/' + filename + '.json', 'w').write(json_string)
	model.save_weights('../models/' + filename + '.h5', overwrite=True)

def save_history(hist, filename):
	"""
	Save the results as a mat file 
	"""
	scipy.io.savemat('../histories/' + 
		filename + '-history.mat', mdict=hist.history)

def save_preds(labels, preds, filename):
	"""
	Save the labels and predictions
	"""
	scipy.io.savemat('../predictions/' + 
		filename + '-preds.mat', mdict={'labels': labels, 'preds': preds})

def save_test_name(test_name):
	"""
	Append the test name to a list of test names
	"""
	with open('../training-code/master-test-list.txt', 'a') as myfile:
		myfile.write(test_name)
		myfile.write('\n')

def get_test_name(user_params):
	"""
	Name the test based on the user params. 
	"""
	test_name = ''
	for key, value in sorted(user_params.items()):
		test_name = test_name + key + '-' + str(value) + '-'
	test_name = test_name.replace(', ', '-')
	exclude = set(string.punctuation.replace('-', ''))
	test_name = ''.join(ch for ch in test_name if ch not in exclude)
	return test_name[0:-1]

def train_lstm(user_params=None):
	"""
	Define default training parameters
	"""
	params = {
			  'lstm_layers': [400],
			  'nepochs': 200,
			  'folds': range(5),
			  'seed': 1337,
			  'optimizer': 'adam',
			  'mask_val': 0,
			  'dropout': 0.5,
			  'max_len': 404,
			  'bsize': 100,
			  'save_model': True,
			  'save_history': True,
			  'save_preds': True,
			  'sensor_data': 'all',
			  'shuffle': True,
			  'verbose': 2
			 }

	test_name = 'default'
	if user_params:
		params.update(user_params)
		test_name = get_test_name(user_params)
	print(test_name)
	save_test_name(test_name)

	# Dropout and lstm layers should be same length
	if numpy.size(params['dropout']) == 1:
		params['dropout'] = [params['dropout']]*len(params['lstm_layers'])
	
	# if seed exists, seed!
	if params['seed']:
		numpy.random.seed(params['seed'])

	"""
	Loop over the folds
	"""
	for fold in params['folds']:
		print('FOLD ' + str(fold) )
		fold_name = '-fold' + str(fold)
		save_name = test_name + fold_name

		print('Loading data...')
		US25, US33, US40, labels, batch_ids = data_loading.load_sensor_data()
		if params['sensor_data'] == 'all':
			stacked = data_loading.concatenate(US25, US33, US40)
		if params['sensor_data'] == 'US25':
			stacked = data_loading.concatenate(US25)
		if params['sensor_data'] == 'US33':
			stacked = data_loading.concatenate(US33)
		if params['sensor_data'] == 'US40':
			stacked = data_loading.concatenate(US40)
		stacked_norm = data_loading.normalize(stacked)
		train_x, train_y, valid_x, valid_y = data_loading.split_5_folds(stacked_norm, labels,
			batch_ids, fold)

		print('Reshaping data...')
		train_x, train_y = data_loading.keras_reshape(train_x, train_y, 
			mask_val=params['mask_val'], max_len=params['max_len'])
		valid_x, valid_y = data_loading.keras_reshape(valid_x, valid_y, 
			mask_val=params['mask_val'], max_len=params['max_len'])

		input_dim = train_x.shape[2]

		print('Building model...')
		model = Sequential()
		model.add(Masking(mask_value=params['mask_val'], 
			input_shape=(params['max_len'], input_dim)))
		for hidden, dropout in zip(params['lstm_layers'][0:-1], params['dropout'][0:-1]):
			model.add(LSTM(output_dim=hidden, return_sequences=True))
			model.add(Dropout(dropout))
			print('Adding layer ' + str(hidden))
		model.add(LSTM(output_dim=params['lstm_layers'][-1]))
		model.add(Dropout(params['dropout'][-1]))
		print('Adding layer ' + str(params['lstm_layers'][-1]))
		model.add(Dense(16)) # was 21
		model.add(Activation('softmax'))
		print('Compiling model...')
		model.compile(loss='categorical_crossentropy',
		        optimizer=params['optimizer'])

		print("Train...")
		hist = model.fit(train_x, train_y,
		        validation_data = (valid_x, valid_y),
		        batch_size=params['bsize'], show_accuracy=True,
		        epochs=params['nepochs'], # previously nb_epoch = params['nepochs']
		        shuffle=params['shuffle'],
		        verbose=params['verbose'])

		# Get predictions on validation set
		output = numpy.asarray(model.predict_on_batch(valid_x))[0, :, :]
		true_labels = numpy.argmax(valid_y, 1)
		preds = numpy.argmax(output, 1)

		# Save model, training history, and predictions
		if params['save_model']:
			print('Saving model...')
			save_model(model, save_name)
		if params['save_history']:
			print('Saving history...')
			save_history(hist, save_name)
		if params['save_preds']:
			print('Saving predictions...')
			save_preds(true_labels, preds, save_name)


def main():
	train_lstm();

if __name__ == "__main__":
	main()

