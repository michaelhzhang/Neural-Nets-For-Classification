import numpy as np 
import scipy as sc
import sys

def clamp(to_clamp):
	"""Helper for avoiding numerical issues. Bounds the value of the input ndarray to avoid
	overflow"""
	return np.clip(to_clamp,-20,20)

def log_clamp(to_clamp):
	"""Helper to avoid numerical issues. Bounds the value of the input array away from zero
	to avoid taking the log of 0"""
	return to_clamp.clip(min=1e-150)

def tanh(vect):
	return 2*sigmoid(2*vect)-1

def tanh_deriv(vect):
	# d/dt tanh t = 1 - tanh^2 t
	result = np.square(tanh(vect))
	result = 1 - result
	return result

def sigmoid(vect):
	# Sigmoid function
	vect = clamp(vect)

	return 1/(1+np.exp(-1*vect))

def sigmoid_deriv(vect):
	# g'(z) = g(z)(1-g(z))
	sigmoided = sigmoid(vect)
	return np.multiply(sigmoided,1-sigmoided)

def ReLU(vect):
	# max(0,x)
	zeros = np.zeros(vect.shape)
	return np.maximum(zeros,vect)

def ReLU_deriv(vect):
	zeros = np.zeros(vect.shape)
	indicators = np.greater(vect,zeros)
	result = 1.0 * indicators # Casting to floats
	return result

def leaky_ReLU(vect):
	# Sets the derivative for low values to be 0.01 instead of 0
	# to fix "dying ReLU problem"
	zeros = np.zeros(vect.shape)
	greater_indicators = np.greater(vect,zeros)
	less_than_indicator = np.logical_not(greater_indicators)

	low_values = 0.01*(1.0*less_than_indicator) # casting to floats
	low_values = np.multiply(vect,low_values)
	high_values = np.multiply(vect,(1.0*greater_indicators))
	result = low_values + high_values
	return result

def leaky_ReLU_deriv(vect):
	# Sets the derivative for low values to be 0.01 instead of 0
	# to fix "dying ReLU problem"
	zeros = np.zeros(vect.shape)
	indicators = np.greater(vect,zeros)
	result = 1.0 * indicators # casting to floats
	lower_bound = zeros + 0.01
	result = np.maximum(result, lower_bound)
	return result

def mean_squared_error(y,z):
	# Note that y and z must be one-hot-encoded
	# z = vector of predicted values
	# Is n x n_out, where n_out = number of classes, n = number of points
	# y = vector of actual values. Is n x n_out
	diff = y - z
	squared = np.square(diff)
	summed = np.sum(squared.flatten())
	return (0.5 * summed)

def cross_entropy(y,z):
	# Note that y and z must be one-hot-encoded
	# z = vector of predicted values
	# Is n x n_out, where n_out = number of classes, n = number of points
	# y = vector of actual values. Is n x n_out
	z = log_clamp(z) # Make sure to avoid numerical errors
	one_minus_z = log_clamp(1-z)
	log_z = np.log(z)
	log_one_minus_z = np.log(one_minus_z)

	first_term = np.multiply(y, log_z)
	second_term = np.multiply(1-y, log_one_minus_z)
	result = first_term + second_term
	result = -1 * result 
	return np.sum(result.flatten())

def mean_squared_error_deriv(y,z):
	# Note that y and z must be one-hot-encoded
	# z = vector of predicted values
	# Is n x n_out, where n_out = number of classes, n = number of points
	# y = vector of actual values. Is n x n_out
	# Output: n_out x n
	return np.transpose(z-y)

def cross_entropy_deriv(y,z):
	# Note that y and z must be one-hot-encoded
	# z = vector of predicted values
	# Is n x n_out, where n_out = number of classes, n = number of points
	# y = vector of actual values. Is n x n_out
	# output: n_out x n
	z = log_clamp(z) # To avoid numerical issues
	one_minus_z = log_clamp(1-z)
	to_subtract = np.divide(y,z)
	result = np.divide(1-y,one_minus_z)
	result = result - to_subtract
	return np.transpose(result)




