import ctypes
import numpy as np
import pandas as pd

import ipdb


class DecisionTree:

	def __init__(self, max_depth : int = 4, min_size : int = 2):
		"""
		Parameters
		----------
		max_depth : maximum depth of the tree
		min_size : minimum size of the tree
		"""
		
		self.max_depth = max_depth
		self.min_size = min_size

	def fit(self, X : pd.DataFrame, y : pd.Series, print_tree=False):
		"""
		Parameters
		----------
		X : Dataframe of features
		y : Series of labels
		"""
		X = pd.concat([X, y], axis=1)
		# we need to have a numpy contiguous array to pass into the C++ fit function
		array = np.ascontiguousarray(X.values[:,:], np.double)

		# ref : https://stackoverflow.com/questions/58727931/how-to-pass-a-2d-array-from-python-to-c
		# create double pointer of type double (cast for 2d array input)
		self.doublePtr = ctypes.POINTER(ctypes.c_double)
		self.doublePtrPtr = ctypes.POINTER(self.doublePtr)

		ct_arr = np.ctypeslib.as_ctypes(array)
		doublePtrArr = self.doublePtr * ct_arr._length_
		ct_ptr = ctypes.cast(doublePtrArr(*(ctypes.cast(row, self.doublePtr) for row in ct_arr)), self.doublePtrPtr)

		# define C++ class initialiser outputs types
		mylib.new_tree.restype = ctypes.c_void_p

		# define input types
		mylib.new_tree.argtypes = [ctypes.c_int, ctypes.c_int]
		mylib.fit_tree.argtypes = [ctypes.c_void_p, self.doublePtrPtr, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
		mylib.predict.argtypes = [ctypes.c_void_p, self.doublePtrPtr, ctypes.c_int, ctypes.c_int]

		self.tree_obj = mylib.new_tree(self.max_depth, self.min_size)
		mylib.fit_tree(self.tree_obj, ct_ptr, array.shape[0], array.shape[1], print_tree)

	def predict(self, X : pd.DataFrame) -> np.array:
		"""
		Parameters
		----------
		X : Dataframe of features

		Returns 
		-------
		predictions : ndarray of shape (n_samples, 1)
		"""
		array = np.ascontiguousarray(X.values[:,:], np.double)

		# define predict outputs types (we need to define it every time to output the 
		# right number of elements we are predicting)
		mylib.predict.restype = ctypes.POINTER(ctypes.c_size_t * array.shape[0])

		ct_arr = np.ctypeslib.as_ctypes(array)

		doublePtrArr = self.doublePtr * ct_arr._length_
		ct_ptr = ctypes.cast(doublePtrArr(*(ctypes.cast(row, self.doublePtr) for row in ct_arr)), self.doublePtrPtr)

		predictions = mylib.predict(self.tree_obj, ct_ptr, array.shape[0], array.shape[1])
		predictions = np.array([i for i in predictions.contents])
		return predictions



# find the shared library compiled after running setup.py
libfile = r"build\lib.win-amd64-3.7\decision_tree.pyd"
mylib = ctypes.CDLL(libfile)

data = pd.read_csv(r"data\setosa.txt", sep="\t")
# ipdb.set_trace()
X , y = data.iloc[:, :-1], data.iloc[:, -1]

# define tree class, fit and predict
dt = DecisionTree(4, 2)
dt.fit(X, y, True)
predictions = dt.predict(X)

print('Predictions: \n{}'.format(predictions))