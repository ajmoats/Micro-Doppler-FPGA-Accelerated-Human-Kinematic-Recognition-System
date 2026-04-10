"""
Data loading functions for Micro-Doppler Recognition
Updated: April 2026
"""

import scipy.io as sio
import numpy
import pickle
import sys

# Path to your MATLAB data file
RAW_DATA_PATH = 'data/data_rot_chunks10.mat'
CHUNKS = 10
ACTORS = 13

def load_sensor_data(sensor='US40'):
	"""
	Load 16/21 actions and a specific sensor frequency.
	Skips actions 6, 7, 8, 9, and 21 (1-indexed).
	Returns data, labels, batch_ids
	"""

	# Load raw data from MATLAB file
	raw_data = sio.loadmat(RAW_DATA_PATH)
	
	# Mapping for 16 actions (0-indexed logic)
	# Original actions (1-21) mapped to 0-indexed indices:
	# We skip indices 5, 6, 7, 8 (Actions 6-9) and 20 (Action 21)
	keep_actions = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
	
	# Select the sensor (US25, US33, or US40)
	sensor_key = 'dataChunks' + sensor
	if sensor_key not in raw_data:
		raise ValueError(f"Sensor {sensor} not found. Options: US25, US33, US40")
	
	sensor_raw = raw_data[sensor_key]
	rand_perm_idx = raw_data['randpermIx'] - 1

	# Temporary lists to hold non-empty sequences
	data_list = []
	labels_list = []
	batch_ids_list = []

	for chunk in range(CHUNKS):
		# new_label_idx (0-15) ensures Keras dense layer compatibility
		for new_label_idx, original_action_idx in enumerate(keep_actions):
			for actor in range(ACTORS):
				idx = (chunk, original_action_idx, actor)
				
				# Check if the sequence exists (is not empty)
				if sensor_raw[idx].size:
					# Transpose from MATLAB to Python format (Time x Features)
					data_list.append(sensor_raw[idx].transpose())
					labels_list.append(new_label_idx)
					batch_ids_list.append(rand_perm_idx[idx])

	# Convert to numpy arrays
	# Use dtype=object because sequences have variable lengths
	data_final = numpy.array(data_list, dtype=object)
	labels_final = numpy.array(labels_list)
	batch_ids_final = numpy.array(batch_ids_list)

	return data_final, labels_final, batch_ids_final

def concatenate(*args):
	"""
	Returns concatenated data (used if you want to combine multiple sensors)
	"""
	return numpy.asarray([numpy.hstack(spectrograms) for spectrograms in zip(*args)])

def normalize(data):
	"""
	Normalize all spectrogram bins to 0 mean, 1 variance
	"""
	all_data = numpy.vstack(data)
	f_means = all_data.mean(0)
	f_stds = all_data.std(0)
	# Prevent division by zero if std is 0
	f_stds[f_stds == 0] = 1.0
	normed_data = numpy.asarray([(seq - f_means) / f_stds for seq in data])

	return normed_data

def split_5_folds(x, y, batch_ids, fold=None):
	"""
	Divide the data into 5 folds for cross-validation
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

def keras_reshape(x, y, mask_val=0, max_len=450, output_dim=16):
	"""
	Prepare data for LSTM input: (nb_samples, max_len, input_dim)
	Output_dim is now 16 to match your filtered actions.
	"""
	nb_samples = len(x)
	input_dim = x[0].shape[1]
	x_reshape = mask_val * numpy.ones((nb_samples, max_len, input_dim))
	y_reshape = numpy.zeros((nb_samples, output_dim))
	
	for sample in range(nb_samples):
		seq = x[sample]
		# Pad or truncate to max_len
		for timestep in range(min(len(seq), max_len)):
			x_reshape[sample, timestep, :] = seq[timestep, :]
		
		# One-hot encode the label
		y_reshape[sample, int(y[sample])] = 1
		
	return x_reshape, y_reshape