import numpy as np
import scipy as sc
import scipy.io
from NeuralNetwork import NeuralNetwork
from NetworkLayer import *
from ActivationAndLossFunctions import *
import matplotlib.pyplot as plt
import cPickle as pickle
from mnist_utils import *

make_snapshots_dir()

# Assumes "train.mat" is the training set from MNIST
train_data = sc.io.loadmat('dataset/train.mat')
train_images = train_data['train_images']
train_labels = train_data['train_labels']

side_length = train_images.shape[0]
preprocessed_images = np.transpose(train_images.reshape((side_length*side_length,-1)))
preprocessed_images = contrast_normalize(preprocessed_images)
training_features, training_labels, validation_features, validation_labels = split_data(preprocessed_images, train_labels, 1/6.0)

test_data = sc.io.loadmat('dataset/test.mat')
test_images = test_data['test_images']
preprocessed_test_images = test_images.reshape((10000, 784))
preprocessed_test_images = contrast_normalize(preprocessed_test_images)

# This actually isn't a great setup for MNIST, multiple hidden layers aren't useful unless you're doing convolutions. 
example_net = NeuralNetwork(cost_func = cross_entropy, cost_deriv = cross_entropy_deriv, 
                                      activation_func = ReLU, activ_deriv = ReLU_deriv,
                                      output_func = softmax, output_deriv = softmax_deriv,
		hid_layer_sizes=[200,200], num_inputs = 784, num_outputs=10, learning_rate = 1e-2, stopping_threshold=-1,
		momentum_rate = 0.9, batch_size = 50, decay_rate = 0.5, decay_frequency = 20,
		cost_calc_freq = 1000, snapshot_frequency = -1,
		snapshot_name = "./snapshots/multilayer_softmax_ReLU", max_iterations = 1e6, relax_targets=False)

example_net.train(training_features, training_labels)

validation_predictions = example_net.predict(validation_features)
benchmark(validation_predictions,validation_labels)
final_predictions = example_net.predict(preprocessed_test_images)
