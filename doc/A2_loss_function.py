import sys
import os
import numpy as np
import scipy.special
from collections import defaultdict
import traceback
from copy import deepcopy

"""

The main purpose of these functions is to calculate the error between the predictions made by a machine learning model and the actual target values.

"""


def _hinge_loss(w, df_new, y_label, C):

    
    yz = y_label * np.dot(df_new,w) # y * (x.w)
    yz = np.maximum(np.zeros_like(yz), (1-yz)) # hinge function
    
    return C*sum(yz)

"""
This function calculates the hinge loss for a linear model with weights w. Hinge loss is commonly used in Support Vector Machines (SVMs). It takes the dot product of input features df_new and weights, multiplies it by the target labels y_label, and then applies the hinge function (max(0, 1 - yz)). The result is scaled by the regularization constant C and summed to return the final hinge loss.

"""


def _logistic_loss(w, X, y, return_arr=None):
	"""Computes the logistic loss.

	This function is used from scikit-learn source code

	Parameters
	----------
	w : ndarray, shape (n_features,) or (n_features + 1,)
	    Coefficient vector.

	X : {array-like, sparse matrix}, shape (n_samples, n_features)
	    Training data.

	y : ndarray, shape (n_samples,)
	    Array of labels.

	"""
	

	yz = y * np.dot(X,w)
	# Logistic loss is the negative of the log of the logistic function.
	if return_arr == True:
		out = -(log_logistic(yz))
	else:
		out = -np.sum(log_logistic(yz))
	return out

"""
This function calculates the logistic loss for a logistic regression model with weights w. It computes the dot product of input features X and weights, multiplies it by the target labels y, and then applies the log-logistic function. The output can be an array or the sum of the array depending on the return_arr parameter.
"""

def _logistic_loss_l2_reg(w, X, y, lam=None):

	if lam is None:
		lam = 1.0

	yz = y * np.dot(X,w)
	# Logistic loss is the negative of the log of the logistic function.
	logistic_loss = -np.sum(log_logistic(yz))
	l2_reg = (float(lam)/2.0) * np.sum([elem*elem for elem in w])
	out = logistic_loss + l2_reg
	return out

"""
This function calculates the logistic loss with L2 regularization for a logistic regression model with weights w. It is similar to the _logistic_loss function but adds an L2 regularization term based on the parameter lam. This helps to prevent overfitting by penalizing large weights.
"""

def log_logistic(X):

	""" This function is used from scikit-learn source code. Source link below """

	"""Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
	This implementation is numerically stable because it splits positive and
	negative values::
	    -log(1 + exp(-x_i))     if x_i > 0
	    x_i - log(1 + exp(x_i)) if x_i <= 0

	Parameters
	----------
	X: array-like, shape (M, N)
	    Argument to the logistic function

	Returns
	-------
	out: array, shape (M, N)
	    Log of the logistic function evaluated at every point in x
	Notes
	-----
	Source code at:
	https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
	-----

	See the blog post describing this implementation:
	http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
	"""
	if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
	out = np.empty_like(X) # same dimensions and data types

	idx = X>0
	out[idx] = -np.log(1.0 + np.exp(-X[idx]))
	out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
	return out

"""
This function computes the log of the logistic function in a numerically stable way. It handles positive and negative values differently to avoid issues with floating-point arithmetic. This implementation is borrowed from scikit-learn's source code and is used within the _logistic_loss function.
"""
