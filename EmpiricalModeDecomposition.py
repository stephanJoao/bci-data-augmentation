import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from Models.DataAugmentation._Generic import generic
from Datasets.CBCIC import CBCIC

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from Models.Preprocessing.EMD import EMD

class EmpiricalModeDecomposition(generic):
	def __init__(self, verbose=False, inplace=False, factor=2):
		super().__init__(verbose, inplace, factor)
		
	def transform(self, X):
		if not self.inplace:
			X = X.copy()			

		# empirical mode decomposition
		decomposed = EMD(resample=None).transform(X)

		# get numpy arrays for X and y
		X_ = X['X']
		y_ = decomposed['y']
		decomposed = decomposed['X']

		# empty lists to store the new trials
		X_new = []
		y_new = []

		# calculate the number of new trials for each class
		n_new_trials = int((self.factor - 1) * X_.shape[0]) // len(np.unique(y_))
		
		# create new trials
		for label in np.unique(y_):
			trials = np.argwhere(y_ == label)[:, 0]

			# X [trials, imfs, electrodes, time]
			X_label = decomposed[trials]

			for _ in range(n_new_trials):
				trials_draw = np.random.randint(0, X_label.shape[0], X_label.shape[1])
				# summation of imfs
				new_trial = []
				for imf in range(X_label.shape[1]):
					new_trial.append(X_label[trials_draw[imf], imf, :, :])
				new_trial = np.array(new_trial).sum(axis=0)
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
	augmenter = EmpiricalModeDecomposition(inplace=True)
	print('Before:')
	print('X: ', data['X'].shape)
	print('y: ', data['y'].shape)
	augmenter.transform(data)
	print('After:')
	print('X: ', data['X'].shape)
	print('y: ', data['y'].shape)