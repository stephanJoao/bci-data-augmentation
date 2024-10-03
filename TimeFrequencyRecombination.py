import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from Models.DataAugmentation._Generic import generic
from Datasets.CBCIC import CBCIC

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft

class TimeFrequencyRecombination(generic):
	def __init__(self, verbose=False, inplace=False, factor=2, fs=128, nperseg=128, noverlap=None, nfft=None, window='hann', return_onesided=True):
		super().__init__(verbose, inplace, factor)
		self.fs = fs
		self.nperseg = nperseg
		self.noverlap = noverlap
		self.nfft = nfft
		self.window = window
		self.return_onesided = return_onesided
		
	def transform(self, X):
		if not self.inplace:
			X = X.copy()			

		# get numpy arrays for X and y
		X_ = X['X']
		y_ = X['y']

		# empty lists to store the new trials
		X_new = []
		y_new = []

		# calculate the number of new trials for each class
		n_new_trials = int((self.factor - 1) * X_.shape[0]) // len(np.unique(y_))
		
		# create new trials
		for label in np.unique(y_):
			trials = np.argwhere(y_ == label)[:, 0]

			# X [trials, electrodes, time]
			X_label = X_[trials]

			# TF [trials, electrodes, freq, time]
			_, _, TF = stft(X_label, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, window=self.window, return_onesided=self.return_onesided)

			# TF [trials, time, electrodes, freq]
			TF = TF.transpose((0, 3, 1, 2))

			for i in range(n_new_trials):
				# get the trials to be combined for each time
				trials_draw = np.random.randint(0, X_label.shape[0], TF.shape[1])
				new_trial = []
				for time in range(TF.shape[1]):
					new_trial.append(TF[trials_draw[time], time, :, :])

				# new_trial [time, electrodes, freq]
				new_trial = np.array(new_trial)

				# new_trial [electrodes, freq, time]
				new_trial = new_trial.transpose((1, 2, 0))

				# istft [electrodes, time]
				_, new_trial = istft(new_trial, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, window=self.window)
				new_trial = new_trial[:, :X_label.shape[2]]
				
				X_new.append(new_trial)
				y_new.append(label)

		# convert to numpy array
		X_new = np.array(X_new)
		y_new = np.array(y_new)

		# concatenate the new trials with the original trials
		X_ = np.concatenate((X_, X_new))
		y_ = np.concatenate((y_, y_new))

		# shuffle the data
		idx = np.random.permutation(len(y_))
		X_, y_ = X_[idx], y_[idx]

		X['X'] = X_
		X['y'] = y_

		if not self.inplace:
			return X
		

if __name__  == '__main__':
	data = CBCIC().load()	
	augmenter = TimeFrequencyRecombination(inplace=True)
	print('Before:')
	print('X: ', data['X'].shape)
	print('y: ', data['y'].shape)
	augmenter.transform(data)
	print('After:')
	print('X: ', data['X'].shape)
	print('y: ', data['y'].shape)