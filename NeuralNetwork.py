import numpy as np 
import scipy as sc 
import numpy.random 
import cPickle as pickle 
import ActivationAndLossFunctions
from NetworkLayer import *
import datetime
from sklearn.preprocessing import OneHotEncoder

class NeuralNetwork:
	def __init__(self, cost_func, cost_deriv, activation_func, activ_deriv, output_func, output_deriv,
		hid_layer_sizes, num_inputs, num_outputs, learning_rate, stopping_threshold = -1, stopping_freq = 10,
		momentum_rate = 0, batch_size = 1, decay_rate = 1, decay_frequency = 1, 
		cost_calc_freq = 1000, snapshot_frequency = -1,
		snapshot_name = "", max_iterations = -1, relax_targets = False, 
		regularization = 0):
		"""
		params: cost_func = J(y,z) where y is actual, z is predicted
		cost_deriv = \partial J/ \partial x
		activation_func = f
		activ_deriv = \partial f/\partial x
		output_func = nonlinearity for output = g
		output_derv = \partial g/ partial x
		hid_layer_sizes = list of ints, with the number of nodes in each hidden layer in order
		num_inputs = number of input features
		num_outputs = number of output nodes
		learning rate for gradient descent
		momentum rate for gradient descent
		decay rate for decaying the learning rate
		decay_frequency = number of epochs after which to decay the learning rate
		batch_size = size of batch for mini-batch
		cost_calc_freq = Number of previous iterations (k) after which to recompute cost to check stopping criteria
		stopping_threshold = Minimum size of l1 norm of all gradients. -1 if not checking
		stopping_freq = Number of iterations of no change to the l1 norm
		snapshot_frequency = number of epochs after which to snapshot this network. -1 if not snapshotting
		snapshot_name = name of this network used during snapshotting. Should include filepath.
		max_iterations = maximum number of gradient descent updates before stopping. -1 if no limit
		relax_targets = True if you want to target 0.85 instead of 1, 0.15 instead of 0 to avoid unit saturation
		regularization = l2 regularization parameter
		"""

		# Initialize hyperparameters
		self.cost_func = cost_func
		self.cost_deriv = cost_deriv
		self.activation_func = activation_func
		self.activ_deriv = activ_deriv
		self.output_func = output_func
		self.output_deriv = output_deriv
		self.hid_layer_sizes = hid_layer_sizes
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.learning_rate = learning_rate
		self.stopping_threshold = stopping_threshold
		self.stopping_freq = stopping_freq
		self.momentum_rate = momentum_rate
		self.batch_size = batch_size 
		self.decay_rate = decay_rate
		self.decay_frequency = decay_frequency
		self.cost_calc_freq = cost_calc_freq
		self.snapshot_frequency = snapshot_frequency
		self.snapshot_name = snapshot_name
		self.max_iterations = max_iterations
		self.relax_targets = relax_targets
		self.regularization = regularization

		# Initialize before training
		self.training_costs = None
		self.training_accuracies = None
		self.num_training_iterations = None 
		self.num_training_epochs = None
		self.decayed_learning_rate = None
		self.total_gradient_size = None

		# construct layers
		self.initialize_layers(cost_func, cost_deriv, activation_func, activ_deriv, output_func, output_deriv,
			hid_layer_sizes, num_inputs, num_outputs)
		# link layers together
		self.connect_layers()
		# Initialize all weights, V, W at random for each layer
		self.initialize_weights()


	def initialize_layers(self,cost_func, cost_deriv, activation_func, activ_deriv, output_func, output_deriv,
		hid_layer_sizes, num_inputs, num_outputs):
		# Helper function. Constructs all the nodes in the network and initializes instance variables
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.num_hidden = len(hid_layer_sizes)
		self.input_layer = InputLayer(num_inputs)
		self.output_layer = OutputLayer(num_outputs, output_func, output_deriv, cost_func, cost_deriv)
		self.hidden_layers = []
		for layer_size in hid_layer_sizes:
			self.hidden_layers.append(HiddenLayer(layer_size,activation_func,activ_deriv))

	def connect_layers(self):
		# Helper function. Constructs the network by linking layers together after they are constructed.
		if (self.num_hidden == 0):
			self.link_layers(self.input_layer,self.output_layer)
		else:
			for i in xrange(self.num_hidden):
				if (i == 0):
					self.link_layers(self.input_layer,self.hidden_layers[i])
				if (i == (self.num_hidden-1)):
					self.link_layers(self.hidden_layers[i],self.output_layer)
				if (0 < i):
					self.link_layers(self.hidden_layers[i-1],self.hidden_layers[i])

	def initialize_weights(self):
		""" Initializes / Resets all the weights in network randomly"""
		self.input_layer.initialize_weights()
		for hid_layer in self.hidden_layers:
			hid_layer.initialize_weights()

	def link_layers(self,layer1, layer2):
		# Helper function. Links 2 layers together
		layer1.set_next_layer(layer2)
		layer2.set_prev_layer(layer1)

	def train(self, images, labels):
		# images: training set (features). Assumes a n x d design matrix
		# where n is number of points, d is number of dimensions.
		# labels: training set (labels). Assumes labels is n_out x 1 numpy array
		num_training = images.shape[0]
		self.reset_training_vars()
		one_hot_encoded_labels = self.one_hot_encode_labels(labels) # n x n_out

		if (self.relax_targets): # For avoiding unit saturation, turn 1s into 0.85 and 0s into 0.15
			one_hot_encoded_labels = self.relax_target_values(one_hot_encoded_labels)
		# Compute initial cost
		initial_cost, initial_accuracy = self.compute_cost_and_accuracy(images,labels,one_hot_encoded_labels)
		self.training_costs = [initial_cost]
		self.training_accuracies = [initial_accuracy]

		# Shuffle data prior to each epoch of training
		shuffled_indices = np.random.permutation(num_training)
		current_index = 0

		# Stopping criterion: Compute total training cost every K iterations
		# If change in cost is too small, then stop
		while (self.check_stopping_criteria(self.training_costs, self.num_training_iterations)):
			self.num_training_iterations += 1
			# In case batch size doesn't divide the number of training examples
			endpoint = min(current_index+self.batch_size, num_training)
			random_batch_indices = shuffled_indices[current_index:endpoint]
			points_to_train = images[random_batch_indices] 
			labels_to_train = one_hot_encoded_labels[random_batch_indices]
			
			# perform forward pass (computing necessary values for gradient descent update)
			self.forward(points_to_train)
			# perform backward pass (again computing necessary values)
			self.backward(labels_to_train)

			self.update_total_gradient_size()
			# Perform gradient descent update
			self.update_weights()

			# Housekeeping
			current_index, shuffled_indices = self.update_training_vars(current_index, shuffled_indices,num_training)
			if (self.check_snapshotting_criteria(current_index)):
				# Periodically save the current network. Do this before computing costs to minimize size of snapshot saved
				self.snapshot()

			if ((self.num_training_iterations > 0) & (self.num_training_iterations % self.cost_calc_freq == 0)):
				# update change in costs for stopping criteria update / for plotting
				cost, accuracy = self.compute_cost_and_accuracy(images,labels,one_hot_encoded_labels)
				self.training_costs.append(cost)
				self.training_accuracies.append(accuracy)
				print "Iteration: " + str(self.num_training_iterations) + ", Accuracy: " + str(self.training_accuracies[-1]) + ", Cost: " + str(self.training_costs[-1])
			

		# Snapshot upon finishing training
		self.snapshot() 

	def add_bias_term(self,data):
		# Assumes data is n x d, where d = number of dimensions, 
		# n = number of points.
		num_points = data.shape[0]
		return np.append(data,np.ones((num_points,1)),axis =1)

	def update_weights(self):
		"""Performs gradient descent update"""
		self.input_layer.update_weights(self.decayed_learning_rate, self.momentum_rate)

	def forward(self, data):
		"""
		Assumes data is n x d, where d = number of dimensions, 
		n = number of points.
		Does forward pass of backprop """
		data = self.input_layer.forward(data)
		for hidden in self.hidden_layers:
			data = hidden.forward(data)
		self.output_layer.forward(data)

	def backward(self, labels):
		"""Does backward pass of backprop
		Assumes labels is one-hot-encoded, n x n_out (n = num points, n_out = number of output values)"""
		error = self.output_layer.backward(labels)
		for hidden in reversed(self.hidden_layers):
			error = hidden.backward(error,self.regularization)
		self.input_layer.backward(error,self.regularization)

	def get_total_gradient_size(self):
		total_gradient = self.input_layer.total_gradient_size()
		for hidden in self.hidden_layers:
			total_gradient += hidden.total_gradient_size()
		return total_gradient

	def update_total_gradient_size(self):
		"""Take the sum total of the size of the gradient over the last stopping_freq iterations"""
		if (self.num_training_iterations > 0) and (self.num_training_iterations % self.stopping_freq == 1):
			self.total_gradient_size = 0
		if (self.total_gradient_size is None):
			self.total_gradient_size = self.get_total_gradient_size()
		else:
			self.total_gradient_size += self.get_total_gradient_size()


	def reset_training_vars(self):
		# Helper for train. Resets training params
		self.reset_all_weight_gradients()
		self.num_training_iterations = 0
		self.num_training_epochs = 0
		self.decayed_learning_rate = self.learning_rate

	def update_training_vars(self, current_index, shuffled_indices, num_training): 
		# Helper for train. Updates counters, decays learning rate and reshuffles data if necessary
		current_index += self.batch_size	
		# increment epochs
		if (current_index >= num_training):
			self.num_training_epochs += 1
			shuffled_indices = np.random.permutation(num_training) # Reshuffle
			current_index = 0
			self.decay_learning_rate()
		return current_index, shuffled_indices

	def get_predicted_probabilities(self, images):
		""" Does a forward pass to generate predictions on images.
		Assumes images is n x d"""
		images = self.input_layer.predict(images)
		for hidden in self.hidden_layers:
			images = hidden.predict(images)
		predicted_probabilities = self.output_layer.predict(images)
		return predicted_probabilities

	def compute_cost_and_accuracy(self, images, labels, one_hot_encoded_labels): 
		# Computes the training error and classification accuracy
		# of this network given images and labels
		predicted_probabilities = self.get_predicted_probabilities(images)	
		predicted_labels = self.generate_labels(predicted_probabilities)
		# Compute cost
		cost = self.cost_func(one_hot_encoded_labels,predicted_probabilities)
		# Compute accuracy
		accuracy = self.classification_accuracy(predicted_labels, labels)

		return cost, accuracy

	def one_hot_encode_labels(self,labels):
		# Assumes labels is a n x 1 vector
		enc = OneHotEncoder(n_values = self.num_outputs)
		encoded_labels = enc.fit_transform(labels)
		return np.array(encoded_labels.todense()) # Change from sparse to dense and convert to ndarray
		# outputs an n x n_out

	def relax_target_values(self,labels):
		# Turns 1s into 0.85, 0s into 0.15. Useful for avoiding unit saturation.
		zeros = np.zeros(labels.shape)
		new_minimum = 0.15 + zeros
		new_maximum = 0.85 + zeros 
		labels = np.minimum(labels, new_maximum)
		labels = np.maximum(labels, new_minimum)
		return labels

	def classification_accuracy(self,predicted,labels):
		# Assumes predicted and labels are both n x 1
		num_points = len(labels) * 1.0
		equality_array = np.equal(predicted,labels).astype(int)
		num_correct = np.sum(equality_array)
		return (num_correct / num_points)

	def check_stopping_criteria(self, training_costs, num_iterations):
		# Checks if gradients are too small or if iteration cap has been reached
		if ((self.max_iterations != -1) and (num_iterations >= self.max_iterations)):
			return False
		if (self.training_accuracies[-1] == 1.0):
			return False # Prevent overfitting
		if (self.stopping_threshold != -1) and (self.total_gradient_size is not None) and (num_iterations % self.stopping_freq == 0):
			return (self.total_gradient_size >= self.stopping_threshold)
		else:
			return True

	def check_snapshotting_criteria(self, current_index):
		if (self.snapshot_frequency == -1):
			return False
		result = ((current_index == 0) and (self.num_training_epochs > 0) and (self.num_training_epochs % self.snapshot_frequency == 0))
		return result 

	def reset_all_weight_gradients(self):
		# Reset the weight gradients of all layers in case we're retraining on a snapshot
		self.input_layer.reset_weight_gradient()
		for hid_layer in self.hidden_layers:
			hid_layer.reset_weight_gradient()
		self.output_layer.reset_weight_gradient()

	def decay_learning_rate(self):
		if (self.num_training_epochs > 0) and (self.num_training_epochs % self.decay_frequency == 0):
			self.decayed_learning_rate = (self.decayed_learning_rate * self.decay_rate)

	def predict(self, images):
		# images: test set (features). 
		# Assumes a n x d design matrix
		# where n is number of points, d is number of dimensions.
		# Returns: Matrix of labels, where each row is the label for point i
		# Run forward pass on network layer
		predictions = self.get_predicted_probabilities(images)
		labels = self.generate_labels(predictions)
		return labels

	def generate_labels(self,prediction):
		# Given matrix of predicted probabilities, returns an array of label vectors
		# Assumes prediction is B X n_out, where n_out = number of output points
		# and n = batch size
		# Outputs an n x 1 vector of labels
		labels = np.argmax(prediction,axis=1)
		# reformat
		return np.transpose([labels])

	def snapshot(self):
		""" Pickles the current network and saves it into directory dir"""
		if (self.snapshot_frequency != -1):
			file_name = self.snapshot_name + "_" 
			file_name = file_name + datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
			file_name = file_name + ".p"
			pickle.dump(self,open(file_name,"wb"))

