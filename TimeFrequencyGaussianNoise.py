import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from Models.DataAugmentation._Generic import generic
from Datasets.CBCIC import CBCIC

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft

class TimeFrequencyGaussianNoise(generic):
	def __init__(self, verbose=False, inplace=False, factor=2, fs=128, nperseg=128, noverlap=None, nfft=None, window='hann', return_onesided=True, std=0.1):
		super().__init__(verbose, inplace, factor)
		self.fs = fs
		self.nperseg = nperseg
		self.noverlap = noverlap
		self.nfft = nfft
		self.window = window
		self.return_onesided = return_onesided
		self.std = std
		
	def transform(self, X):
		if not self.inplace:
			X = X.copy()			

		# get numpy arrays for X and y
		X_ = X['X']
		y_ = X['y']

		# calculate the number of new trials for each class
		n_new_trials = int((self.factor - 1) * X_.shape[0]) // len(np.unique(y_))
		
		# create new trials
		for label in np.unique(y_):
			trials = np.argwhere(y_ == label)[:, 0]

			# X [trials, electrodes, time]
			X_label = X_[trials]

			# TF [trials, electrodes, freq, time]
			_, _, TF = stft(X_label, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, window=self.window, return_onesided=self.return_onesided)

			# get the drawn	trials
			trials_draw = np.random.randint(0, TF.shape[0], n_new_trials)
			TF = TF[trials_draw]

			# separate amplitude and phase
			A = abs(TF)
			phi = np.angle(TF)

			# add gaussian noise to the amplitude
			A = A + np.random.normal(0, 0.1 * np.std(A), A.shape)

			# combine amplitude and phase in new TF
			TF = A * np.exp(1j * phi)

			_, X_new = istft(TF, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft, window=self.window)
			X_new = X_new[:, :, :X_label.shape[2]]
			# add n_new_trials 
			y_new = [label] * n_new_trials

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
	augmenter = TimeFrequencyGaussianNoise(inplace=True)
	print('Before:')
	print('X: ', data['X'].shape)
	print('y: ', data['y'].shape)
	augmenter.transform(data)
	print('After:')
	print('X: ', data['X'].shape)
	print('y: ', data['y'].shape)