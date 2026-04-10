"""
Data loading functions
"""

import scipy.io as sio
import numpy
#import cPickle as pickle
import pickle
import sys

#RAW_DATA_PATH = '../data/data_rot_chunks10.mat'
RAW_DATA_PATH = 'data/data_rot_chunks10.mat'
CHUNKS = 10
ACTIONS = 21
ACTORS = 13


def load_sensor_data():
	"""
	Load the sensor spectrogram data from the .mat file
	Return US25_data, US33_data, US40_data, labels, batch_ids

	Data is flattened into an ndarray containing ndarrays of spectrograms of action sequences.
	"""

	# Load raw data into individual sensor files
	raw_data = sio.loadmat(RAW_DATA_PATH)
	US25_data = raw_data['dataChunksUS25'][:, 0:ACTIONS, :]
	US33_data = raw_data['dataChunksUS33'][:, 0:ACTIONS, :]
	US40_data = raw_data['dataChunksUS40'][:, 0:ACTIONS, :]
	rand_perm_idx = raw_data['randpermIx'] - 1

	# Transpose the sensor data, create labels and batch ids. Empty sequences converted to None
	US25_data_T = numpy.empty_like(US25_data)
	US33_data_T = numpy.empty_like(US33_data)
	US40_data_T = numpy.empty_like(US40_data)
	labels = numpy.empty_like(US25_data)
	batch_ids = numpy.empty_like(US25_data)
	for chunk in range(CHUNKS):
		for action in range(ACTIONS):
			for actor in range(ACTORS):
				idx = (chunk, action, actor)
				# Check for empty sequences
				if US25_data[idx].size:
					US25_data_T[idx] = US25_data[idx].transpose()
					US33_data_T[idx] = US33_data[idx].transpose()
					US40_data_T[idx] = US40_data[idx].transpose()
					labels[idx] = action
					batch_ids[idx] = rand_perm_idx[idx]

	# Flatten relo
	US25 = US25_data_T.flatten()
	US33 = US33_data_T.flatten()
	US40 = US40_data_T.flatten()
	labels = labels.flatten()
	batch_ids = batch_ids.flatten()

	# Find valid sequence ids
	non_empty_ids = numpy.where(numpy.not_equal(labels, None))
	return US25[non_empty_ids], US33[non_empty_ids], US40[non_empty_ids], \
		labels[non_empty_ids], batch_ids[non_empty_ids]

def concatenate(*args):
	"""
	Returns concatenated data

	args :: list of sensor data to be concatenated
	"""
	return numpy.asarray([numpy.hstack(spectrograms) for spectrograms in zip(*args)])

def normalize(data):
	"""
	normalize all spectrogram bins to 0 mean, 1 variance
	"""
	all_data = numpy.vstack(data)
	f_means = all_data.mean(0)
	f_stds = all_data.std(0)
	normed_data = numpy.asarray([(seq - f_means) / f_stds for seq in data])

	return normed_data

def split_5_folds(x, y, batch_ids, fold=None):
	"""
	Divide the data into 5 folds like Tom did

	Returns either a specific fold or a list of all folds if none is specified
	"""
	nchunks = 10
	nbatches = 5
	ntrain = 8
	ntest = nchunks - ntrain

	if fold == None:
		folds = []

		for bb in range(nbatches):
			test_batches = numpy.arange(ntest) + bb * ntest
			train_batches = numpy.delete(numpy.arange(nchunks), test_batches)

			test_idx = numpy.where(numpy.in1d(batch_ids, test_batches))
			train_idx = numpy.where(numpy.in1d(batch_ids, train_batches))

			folds.append((x[train_idx], y[train_idx], x[test_idx], y[test_idx]))

		return folds
	else:
		test_batches = numpy.arange(ntest) + fold * ntest
		train_batches = numpy.delete(numpy.arange(nchunks), test_batches)

		test_idx = numpy.where(numpy.in1d(batch_ids, test_batches))
		train_idx = numpy.where(numpy.in1d(batch_ids, train_batches))

		return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def split_MCCV(x, y, ntest=100):
	"""
	Split the data for Monte Carlo cross-validation. 
	Train and test sets of size nexamples - ntest and ntest, respectively

	x :: ndarray of sequences
	y :: corresponding ndarray of labels
	"""

	test_idx = numpy.random.choice(x.shape[0], ntest, replace=False)
	train_idx = numpy.delete(numpy.arange(x.shape[0]), test_idx)

	train_x = x[train_idx]
	train_y = y[train_idx]
	test_x = x[test_idx]
	test_y = y[test_idx]

	return train_x, train_y, test_x, test_y

def keras_reshape(x, y, mask_val=0, max_len=450, output_dim=21):
	"""
	Generate a reshaped version of x, initialize with 10's which will be masked
	by masking layer. Use 10's as data is normalized to N(0,1)
	Maximum sequence length is about 404 timesteps. max_len defaults above that at 450
	
	x_reshape :: (nb_samples, max_len, input_dim)
	y_reshape :: (nb_samples, max_len, output_dim)
	"""
	nb_samples = len(x)
	input_dim = x[0].shape[1]
	x_reshape = mask_val * numpy.ones((nb_samples, max_len, input_dim))
	y_reshape = numpy.zeros((nb_samples, output_dim))
	
	# Loop over each sequence
	for sample in range(nb_samples):
		# Create x sequences
		seq = x[sample]
		for timestep in range(min(len(seq), max_len)):
		    x_reshape[sample, timestep, :] = seq[timestep, :]
		# Create y sequences
		y_reshape[sample, y[sample]] = 1
	return x_reshape, y_reshape
