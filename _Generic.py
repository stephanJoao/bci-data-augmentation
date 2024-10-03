import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

class generic:    
    # (parameter) factor: factor for augmentation, 
    # e.g. if factor = 2, the new size is 2 times bigger then the input
    def __init__(self, verbose=False, inplace=False, factor=2):
        self.verbose = verbose
        self.inplace = inplace
        self.factor = factor

    # (parameter) X: data as mne.epochs
    # (exit) self
    # (description) This function do nothing, it's just for compatibility problems
    def fit(self, X):
        return self
    
    # (parameter) X: data as mne.epochs
    # (exit) augmented data as mne.epochs
    # (description) This function create artificial data to increase the data size
    def transform(self, X):
        return X

    # (parameter) X: data as mne.epochs
    # (exit) augmented data as mne.epochs
    # (description) This function calls the functions fit and transform
    def fit_transform(self, X):
        self.fit(X)
        newX = self.transform(X)
        return newX
    
    # (parameters) deep: inform if the parameters will be returned together with the hyperparameters
    # (exit) dictionary with the variables of the class
    def get_params(self, deep):
        return {'verbose': self.verbose, 'inplace': self.inplace, 'factor': self.factor}