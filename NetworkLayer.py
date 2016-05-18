import numpy as np 
import scipy as sc 
import numpy.random 
import math 
import ActivationAndLossFunctions
import sys

class Layer:
	def __init__(self, num_nodes):
		# prev_layer, next_layer: references
		# num_nodes: number of nodes in current layer, NOT counting the bias
		# prev_num_nodes: number of nodes in previous layer, NOT counting the bias
		# next_num_nodes: number of nodes in next layer, NOT counting the bias
		self.prev_layer = None
		self.next_layer = None
		self.num_nodes = num_nodes 
		self.prev_num_nodes = None
		self.next_num_nodes = None
		self.weights = None
		self.last_weight_update = None # for momentum
		self.outputs = None

	def set_prev_layer(self, prev_layer):
		# Sets prev_layer
		self.prev_layer = prev_layer
		self.prev_num_nodes = prev_layer.get_num_nodes()

	def set_next_layer(self, next_layer):
		self.next_layer = next_layer
		self.next_num_nodes = next_layer.get_num_nodes()

	def get_num_nodes(self):
		return self.num_nodes

	def initialize_weights(self):
		"""Initialize input weights with normal distribution, centered at 0 w/ std 0.01"""
		assert (self.next_num_nodes is not None)
		assert (self.num_nodes is not None)
		# Gaussian centered at 0 with standard deviation 0.01
		self.weights = 0.01 * np.random.randn(self.num_nodes + 1,self.next_num_nodes)

	def reset_weight_gradient(self):
		""" Reset weight gradients for the next pass, and frees up unneeded data. """
		if (self.weights is None):
			self.total_weight_gradient = None
		else:
			self.total_weight_gradient = 0

		self.outputs = None 

	def update_weights(self, learning_rate, momentum_rate):
		if (self.weights is not None):
			adjustment = (learning_rate)*(self.weight_gradient)
			if (self.last_weight_update is None):
				self.last_weight_update = -1*adjustment
			else:
				self.last_weight_update = (-1*adjustment) + (momentum_rate * self.last_weight_update)
			self.weights = self.weights + self.last_weight_update
			self.reset_weight_gradient() # Have to reset for the next pass
		if (self.next_layer is not None): # Update the next layer
			self.next_layer.update_weights(learning_rate,momentum_rate)

	def l1_norm(self,gradient):
		# Computes l1 norm of gradient updates
	    result = np.abs(gradient)
	    result = np.sum(result)
	    return result

	def add_bias_term(self,data):
		# Assumes data is n x d, where d = number of dimensions, 
		# n = number of points.
		num_points = data.shape[0]
		return np.append(data,np.ones((num_points,1)),axis =1)


class InputLayer(Layer):
	# Layer stores the OUTPUT weights, not the input weights
	def __init__(self, num_nodes):
		Layer.__init__(self,num_nodes)

	def __str__(self):
		return "InputLayer"

	def forward(self, data):
		# Assumes data is n x d design matrix
		self.outputs = self.add_bias_term(data)
		next_input = np.dot(self.outputs,self.weights)
		return next_input

	def backward(self, prev_error,regularization):
		self.weight_gradient = np.transpose(np.dot(prev_error,self.outputs))
		if (regularization != 0):
			self.weight_gradient += (regularization*self.weights)

	def predict(self, data):
		result = np.dot(self.add_bias_term(data),self.weights)
		return result

	def total_gradient_size(self):
		"""Returns L1 norm of the last gradient"""
		total = self.l1_norm(self.total_weight_gradient)
		return total


class HiddenLayer(Layer):
	# Layer stores the OUTPUT weights, not the input weights
	def __init__(self, num_nodes, activation_func, activation_deriv):
		Layer.__init__(self,num_nodes)
		self.activation_func = activation_func
		self.activation_deriv = activation_deriv

	def __str__(self):
		return "HiddenLayer"

	def forward(self, inputs):
		# Assumes inputs is an n x d by design matrix
		# Compute and cache new outputs
		self.outputs = self.add_bias_term(self.activation_func(inputs)) # f(s_i)
		next_input = np.dot(self.outputs, self.weights)
		self.activation_derivatives = np.transpose(self.activation_deriv(inputs)) # f'(s_i)^T
		return next_input

	def predict(self, inputs): 
		# Used for predictions, without caching results needed for gradient descent
		result = self.activation_func(inputs)
		result = np.dot(self.add_bias_term(result), self.weights)
		return result

	def backward(self, prev_error, regularization):
		# Compute gradients with respect to weight
		weights_no_bias = self.weights[0:-1,] # Don't compute errors for bias
		error = np.dot(weights_no_bias,prev_error) 
		error = np.multiply(self.activation_derivatives, error) 
		self.weight_gradient = np.transpose(np.dot(prev_error,self.outputs))
		if (regularization != 0):
			self.weight_gradient += (regularization*self.weights)

		return error

	def total_gradient_size(self):
		"""Returns L1 norm of the last gradient"""
		total = self.l1_norm(self.total_weight_gradient)
		return total

class OutputLayer(Layer):
	# Layer stores the OUTPUT weights, not the input weights
	def __init__(self,num_nodes, activation_func, activation_deriv, cost_func, cost_deriv):
		Layer.__init__(self,num_nodes)
		self.activation_func = activation_func
		self.activation_deriv = activation_deriv
		self.cost_func = cost_func
		self.cost_deriv = cost_deriv

	def __str__(self):
		return "OutputLayer"

	def forward(self,inputs):
		# Assumes prev_output is n x d, where d = number of dimensions, 
		# n = number of points.
		self.outputs = self.activation_func(inputs) # f(s_i)
		self.activation_derivatives = np.transpose(self.activation_deriv(inputs)) # f'(s_i)^T

	def predict(self, inputs): 
		# Used for predictions, without caching results needed for gradient descent
		result = self.activation_func(inputs) #f(s_i) B x n_out
		return result

	def backward(self,labels):
		# Assumes labels is one-hot-encoded, n x n_out (n = num points, n_out = number of output values)
		# Output: n_out x n
		error = self.cost_deriv(labels,self.outputs)
		error = np.multiply(self.activation_derivatives,error)

		return error

