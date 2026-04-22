import LSTM.Original_Code.JhummaLstm as JhummaLstm

# Run with default params.
# JhummaLstm.train_lstm()

# Run with 2 layers
# params = {
# 	'lstm_layers': [400, 400]
# 	}
# JhummaLstm.train_lstm(params)

# Run with one large layer
# params = {
# 	'lstm_layers': [800]
# 	}
# JhummaLstm.train_lstm(params)

# # Run with one large layer
# params = {
# 	'lstm_layers': [1200]
# 	}
# JhummaLstm.train_lstm(params)

# # Run with one large layer
# params = {
# 	'lstm_layers': [800, 800]
# 	}
# JhummaLstm.train_lstm(params)

# # Run with one large layer
# params = {
# 	'lstm_layers': [800, 400]
# 	}
# JhummaLstm.train_lstm(params)

# Run with one large layer
# params = {
# 	'lstm_layers': [1600]
# 	}
# JhummaLstm.train_lstm(params)

# Run with one medium layer with shuffle=True
# params = {
# 	'lstm_layers': [800],
# 	'shuffle': True
# 	}
# JhummaLstm.train_lstm(params)

# Run with small batch size
# params = {
# 	'lstm_layers': [400],
# 	'bsize': 20,
# 	'nepochs': 300
# 	}
# JhummaLstm.train_lstm(params)

# Run with medium batch size
params = {
	'lstm_layers': [400],
	'bsize': 50,
	}
JhummaLstm.train_lstm(params)

# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python test_run.py