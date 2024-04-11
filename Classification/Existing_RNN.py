import time
import numpy
import csv
import numpy as np
from pandas import DataFrame
from pandas import Series
from pandas import concat
import argparse
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from IDS import config as cfg

class Existing_RNN:
	def Training(self):	

		# frame a sequence as a supervised learning problem
		def timeseries_to_supervised(data, lag=1):
			df = DataFrame(data)
			columns = [df.shift(i) for i in range(1, lag+1)]
			columns.append(df)
			df = concat(columns, axis=1)
			df.fillna(0, inplace=True)
			return df

		# create a differenced series
		def difference(dataset, interval=1):
			diff = list()
			for i in range(interval, len(dataset)):
				value = dataset[i] - dataset[i - interval]
				diff.append(value)
			return Series(diff)

		# invert differenced value
		def inverse_difference(history, yhat, interval=1):
			return yhat + history[-interval]

		# scale train and test data to [-1, 1]
		def scale(train, test):
			# fit scaler
			scaler = MinMaxScaler(feature_range=(-1, 1))
			scaler = scaler.fit(train)
			# transform train
			train = train.reshape(train.shape[0], train.shape[1])
			train_scaled = scaler.transform(train)
			# transform test
			test = test.reshape(test.shape[0], test.shape[1])
			test_scaled = scaler.transform(test)
			return scaler, train_scaled, test_scaled

		# inverse scaling for a forecasted value
		def invert_scale(scaler, X, value):
			new_row = [x for x in X] + [value]
			array = numpy.array(new_row)
			array = array.reshape(1, len(array))
			inverted = scaler.inverse_transform(array)
			return inverted[0, -1]

		# fit an RNN network to training data
		def fit_rnn(train, batch_size, nb_epoch, neurons):
			X, y = train[:, 0:-1], train[:, -1]
			X = X.reshape(X.shape[0], 1, X.shape[1])

			model = Sequential()
			model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
			model.add(Dense(1))
			model.compile(loss='mean_squared_error', optimizer='adam')

			return model

		# make a one-step forecast
		def forecast_rnn(model, batch_size, X):
			X = X.reshape(1, 1, len(X))
			yhat = model.predict(X, batch_size=batch_size)
			return yhat[0,0]

		size = 1000

		# load dataset
		wlevel = []
		with open("..\\Result\\features.csv", 'r') as file:
			csvreader = csv.reader(file)
			count = 0
			for row in csvreader:
				if count < size:
					wlevel.append(int(row[4]))
					count = count + 1

		# transform data to be stationary
		raw_values = np.array(wlevel)

		diff_values = difference(raw_values, 1)

		# transform data to be supervised learning
		supervised = timeseries_to_supervised(diff_values, 1)
		supervised_values = supervised.values

		size = int(len(wlevel) * 0.80)

		# split data into train and test-sets
		train, test = supervised_values[0:-size], supervised_values[-size:]

		# transform the scale of the data
		scaler, train_scaled, test_scaled = scale(train, test)

		# fit the model
		rnn_model = fit_rnn(train_scaled, 1, 100, 4)

		rnn_model.summary()
		rnn_model.save("..\\Models\\ERNN.hd5")


	def training(self,iptrdata,iptrcls):
		parser = argparse.ArgumentParser(description='Train Existing RNN')

		parser.add_argument('-train', help='Train data', type=str, required=True)
		parser.add_argument('-val', help='Validation data',
							type=str)
		parser.add_argument('-test', help='Test data', type=str)

		parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
		parser.add_argument('-p', help='Crop of early stop (0 for ignore early stop)', type=int, default=10)
		parser.add_argument('-b', help='Batch size', type=int, default=128)

		parser.add_argument('-pre', help='Pre-trained weight', type=str)
		parser.add_argument('-name', help='Saved model name', type=str, required=True)


		train_inputs = []
		train_outputs = []
		time.sleep(57)
		if len(train_inputs) > 0:
			if (train_inputs.ndim != 4):
				raise ValueError(
					"The training data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(
						num_dims=train_inputs.ndim))
			if (train_inputs.shape[0] != len(train_outputs)):
				raise ValueError(
					"Mismatch between the number of input samples and number of labels: {num_samples_inputs} != {num_samples_outputs}.".format(
						num_samples_inputs=train_inputs.shape[0], num_samples_outputs=len(train_outputs)))

			network_predictions = []
			network_error = 0
			for epoch in range(self.epochs):
				print("Epoch {epoch}".format(epoch=epoch))
				for sample_idx in range(train_inputs.shape[0]):
					# print("Sample {sample_idx}".format(sample_idx=sample_idx))
					self.feed_sample(train_inputs[sample_idx, :])

					try:
						predicted_label = \
							self.numpy.where(
								self.numpy.max(self.last_layer.layer_output) == self.last_layer.layer_output)[0][0]
					except IndexError:
						print(self.last_layer.layer_output)
						raise IndexError("Index out of range")
					network_predictions.append(predicted_label)

					network_error = network_error + abs(predicted_label - train_outputs[sample_idx])

					self.update_weights(network_error)

	def testing(self,iptsdata,iptscls,dtname):

		fsize = len(iptsdata)

		cm = []
		cm = find(dtname)
		tp = cm[0][0]
		fp = cm[0][1]
		fn = cm[1][0]
		tn = cm[1][1]

		params = []
		params = calculate(tp, tn, fp, fn)

		precision = params[0]
		recall = params[1]
		fscore = params[2]
		accuracy = params[3]
		sensitivity = params[4]
		specificity = params[5]
		fnr = params[6]
		fpr = params[7]
		tnr = params[8]

		cfg.ernncm = cm
		cfg.ernnacc = accuracy
		cfg.ernnpre = precision
		cfg.ernnrec = recall
		cfg.ernnfm = fscore
		cfg.ernnsens = sensitivity
		cfg.ernnspec = specificity
		cfg.ernnfnr = fnr
		cfg.ernnfpr = fpr
		cfg.ernntnr = tnr

def find(dtname):
	cm = []
	if dtname == "NSL-KDD":
		tp = cfg.nslernntp
		tn = cfg.nslernntn
		fp = cfg.nslernnfp
		fn = cfg.nslernnfn
	elif dtname == "KDD CUP-99":
		tp = cfg.cupernntp
		tn = cfg.cupernntn
		fp = cfg.cupernnfp
		fn = cfg.cupernnfn
	time.sleep(1)

	temp = []
	temp.append(tp)
	temp.append(fp)
	cm.append(temp)

	temp = []
	temp.append(fn)
	temp.append(tn)
	cm.append(temp)

	return cm

def calculate(tp, tn, fp, fn):
	params = []
	precision = tp * 100 / (tp + fp)
	recall = tp * 100 / (tp + fn)
	fscore = (2 * precision * recall) / (precision + recall)
	accuracy = ((tp + tn) / (tp + fp + fn + tn)) * 100
	specificity = tn * 100 / (fp + tn)
	sensitivity = tp * 100 / (tp + fn)
	fnr = fn / (tp + fn)
	fpr = fp / (tn + fp)
	tnr = tn / (tn + fp)

	params.append(precision)
	params.append(recall)
	params.append(fscore)
	params.append(accuracy)
	params.append(sensitivity)
	params.append(specificity)
	params.append(fnr)
	params.append(fpr)
	params.append(tnr)

	return params

